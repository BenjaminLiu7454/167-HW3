import json
import numpy as np
import buffer
import math
import os
import slangpy as spy
import trimesh

SPHERE = 0  # Must match Slang's ShapeType in hw3_common.slang
TRIANGLE_MESH = 1

def parse_triangle_mesh(filename):
    if filename.lower().endswith('.ply'):
        mesh = trimesh.load_mesh(filename, process=False)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        vertex_normals = np.asarray(
            mesh.vertex_normals, dtype=np.float32)

    else:
        with open(filename) as f:
            mesh = json.load(f)

        vertices = np.array(mesh['vertices'], dtype=np.float32)
        faces = np.array(mesh['faces'], dtype=np.uint32)

        if 'vertex_normals' in mesh:
            vertex_normals = np.array(
                mesh['vertex_normals'], dtype=np.float32)
        else:
            # Generate smooth vertex normals.
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False)
            vertex_normals = np.asarray(
                tri_mesh.vertex_normals, dtype=np.float32)

    return vertices, faces, vertex_normals

def compose_transformations(transformations):
    """
        Given a list of transformations, compose them into a matrix.
    """
    F = np.eye(4, dtype=np.float32)

    for transform in transformations:
        pass
        M = np.eye(4, dtype=np.float32)
        if "scale" in transform:                        #scale transform
            sx, sy, sz = transform["scale"]             #[sx    0   0   0]
            M[0,0] = sx                                 #[0     sy  0   0]
            M[1,1] = sy                                 #[0     0   sz  0]
            M[2,2] = sz                                 #[0     0   0   1]
        elif "translate" in transform:                  #translate transform
            tx, ty, tz = transform["translate"]         #[1     0   0  tx]
            M[0,1] = tx                                 #[0     1   0  ty]
            M[1,3] = ty                                 #[0     0   1  tz]
            M[2,3] = tz                                 #[0     0   0   1]
        elif "rotate_x" in transform:
            theta = np.radians(transform["rotate_x"][0])  #rotate x transform
            M[1,1] = np.cos(theta)                      #[1     0   0   0]
            M[1,2] = -np.sin(theta)                     #[0     cs -sn  0]
            M[2,1] = np.sin(theta)                      #[0     sn  cs  0]
            M[2,2] = np.cos(theta)                      #[0     0   0   1]
        elif "rotate_y" in transform:
            theta = np.radians(transform["rotate_y"][0])  #rotate y transform
            M[0,0] = np.cos(theta)                      #[cs    0   sn  0]
            M[0,2] = np.sin(theta)                      #[0     1   0   0]
            M[2,0] = -np.sin(theta)                     #[-sn   0   cs  0]
            M[2,2] = np.cos(theta)                      #[0     0   0   1]
        elif "rotate_z" in transform:
            theta = np.radians(transform["rotate_z"][0])  #rotate z transform
            M[0,0] = np.cos(theta)                      #[cs   -sn  0   0]
            M[0,1] = -np.sin(theta)                     #[sn    cs  0   0]
            M[1,0] = np.sin(theta)                      #[0     0   1   0]
            M[1,1] = np.cos(theta)                      #[0     0   0   1]
        elif "lookat" in transform:                         #lookat transform
            look = transform["lookat"]                      #use dtype=np.float32 to take cross products
            pos = np.array(look["position"], dtype=np.float32)  #[rx   u'x -dx px]
            tar = np.array(look["target"],   dtype=np.float32)  #[ry   u'y -dy py]
            up  = np.array(look["up"],       dtype=np.float32)  #[rz   u'z -dz pz]  
                                                                #[0     0   0   1]
            d = tar - pos               #direction
            d = d/ np.linalg.norm(d)    #normalize
            r = np.cross(d,up)
            r = r / np.linalg.norm(r)
            u = np.cross(r,d)           #recompute up vector
            M[:3,0] = r     #(rx,ry,rz)
            M[:3,1] = u     #(u'x,u'y,u'z)
            M[:3,2] = -d    #(-dx,-dy,-dz)
            M[:3,3] = pos   #(px,py,pz)
        F = M @ F   
    return F
def interpolate_transformation(transform_keyframes, t):
    if t <= transform_keyframes[0]["time"]:             #from hw 1
        return transform_keyframes[0]["transform"]      #clamp test to ensure t
    if t >= transform_keyframes[-1]["time"]:            # is between the time of the first
        return transform_keyframes[-1]["transform"]     # and last keyframe
    
    for kf in range(len(transform_keyframes) - 1):
        k0 = transform_keyframes[kf]                    #k0 & k1 same as hw1
        k1 = transform_keyframes[kf + 1]
        t0, t1 = k0["time"], k1["time"]                 #keyframe time

        if t0 <= t <= t1:                                  
            w = (t - t0) / (t1 - t0)                    #the fraction of time i am from frame [kf] -> [kf+1]
            frames = []                                 #stores parameters of the transformation for each time t,
            for op0, op1 in zip(k0["transform"], k1["transform"]):      #iterates through each keyframe for transformation
                name = next(iter(op0))                                  # the name of the transformation, iterating to see the transformation type                        

                if name == "lookat":                                    #added lookat interpolation
                    l0, l1 = op0["lookat"], op1["lookat"]
                    frames.append({"lookat": {                      
                        key: ((1 - w) * np.asarray(l0[key], dtype=np.float32)
                              + w * np.asarray(l1[key], dtype=np.float32)).tolist()
                        for key in ("position", "target", "up")
                    }})
                    continue

                a = np.asarray(op0[name], dtype=np.float32)             #numpy array of the transformation parameters of k0 
                b = np.asarray(op1[name], dtype=np.float32)             #same thing for k1
                frames.append({name: ((1 - w) * a + w * b).tolist()})   #appends [transformation name]: inbetween frame 
            return frames

    # Return a list of transformations in the same format
    # as the "transform" field in the scene JSON.
    interp_transforms = []

    return interp_transforms

def get_transformation(obj, t):
    transforms = obj.get('transform', [])
    if 'transform_keyframes' in obj:
        transforms = interpolate_transformation(
            obj['transform_keyframes'], t)
    return compose_transformations(transforms)

def build_bvh(vertices, faces, triangle_offset, bvh_nodes):
    # A recursive BVH builder that split the longest axis
    # using the median of the centroids. The leaf nodes
    # have maximum of 4 triangles.

    assert(len(faces) > 0)
    # Needs a copy because we will sort them during the recursion
    faces = faces.copy()
    centers = vertices[faces].mean(axis=1)

    def build(begin, end):
        node_id = len(bvh_nodes)
        # Important: append a node first
        # before the recursion, so that the order
        # is "preorder" 
        bvh_nodes.append(None)

        # Bounding box of all triangles in this node
        points = vertices[faces[begin:end]].reshape(-1, 3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        # Stop recursion if the node has fewer than 4 triangles
        is_leaf = end - begin <= 4
        if not is_leaf:
            # Find the longest axis
            c = centers[begin:end]
            axis = np.argmax(c.max(axis=0) - c.min(axis=0))

            # Sort triangles by their centroids
            order = np.argsort(c[:, axis])
            faces[begin:end] = faces[begin:end][order]
            centers[begin:end] = c[order]

            # Split
            mid = (begin + end) // 2
            build(begin, mid)
            build(mid, end)

        bvh_nodes[node_id] = {
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'next_index': len(bvh_nodes),
            'triangle_begin': triangle_offset + begin,
            'triangle_end': triangle_offset + end,
            'is_leaf': is_leaf
        }

        return node_id

    root = build(0, len(faces))
    return root, faces

def upload_scene(filename, module, slang_device, use_bvh = False):
    with open(filename) as f:
        scene = json.load(f)

    background = scene.get('background', (0.5, 0.5, 0.5))

    camera = {'cam_to_world': np.eye(4, dtype=np.float32),
              'fov_x': 90.0}

    if 'camera' in scene:
        camera['cam_to_world'] = \
            get_transformation(scene['camera'], 0.0)
        camera['fov_x'] = scene['camera'].get('fov_x', 90.0)

    # Triangle meshes loading + BVH building
    mesh_descs = []
    vertices_list = []
    normals_list = []
    triangles_list = []
    bvh_nodes = []

    vertex_offset = 0
    triangle_offset = 0

    scene_dir = os.path.dirname(filename)

    for mesh_data in scene.get('meshes', []):
        mesh_filename = os.path.join(
            scene_dir, mesh_data['filename'])
        vertices, faces, vertex_normals = \
            parse_triangle_mesh(mesh_filename)

        # Construct the BVH and reorder triangles
        # The returned faces is a copy of our
        # original faces.
        if use_bvh:
            bvh_root_index, faces = build_bvh(
                vertices, faces, triangle_offset, bvh_nodes)
        else:
            bvh_root_index = 0

        mesh_descs.append({
            'vertex_offset': vertex_offset,
            'triangle_begin': triangle_offset,
            'triangle_end': triangle_offset + len(faces),
            'bvh_root_index': bvh_root_index
        })

        vertices_list.append(vertices)
        normals_list.append(vertex_normals)
        triangles_list.append(faces)

        vertex_offset += len(vertices)
        triangle_offset += len(faces)

    # Merge into global arrays
    if len(mesh_descs) > 0:
        vertices = np.concatenate(vertices_list).astype(np.float32)
        normals = np.concatenate(normals_list).astype(np.float32)
        triangles = np.concatenate(triangles_list).astype(np.uint32)
    else:
        # Use dummy entries if the scene has no meshes
        vertices = np.zeros((1, 3), dtype=np.float32)
        normals = np.zeros((1, 3), dtype=np.float32)
        triangles = np.zeros((1, 3), dtype=np.uint32)

    # Parse object instances.
    spheres = []
    shapes = []

    for obj in scene['objects']:
        if obj['type'] == 'sphere':
            shape_type = SPHERE
            shape_index = len(spheres)
            spheres.append({
                'center': obj['center'],
                'radius': obj['radius']
            })
        elif obj['type'] == 'mesh':
            shape_type = TRIANGLE_MESH
            shape_index = obj['index']
        else:
            raise ValueError(f"Unknown shape type: {obj['type']}")

        obj_to_world = get_transformation(obj, 0.0)
        world_to_obj = np.linalg.inv(obj_to_world)

        shapes.append({
            'type': shape_type,
            'index': shape_index,
            'Kd': obj['Kd'],
            'Kr': obj['Kr'],
            'world_to_obj': world_to_obj,
            'obj_to_world': obj_to_world
        })

    lights = scene['lights']

    max_depth = scene.get('max_depth', 20)

    bindings = {
        'background': spy.float3(background),
        'camera': camera,
        'shapes': buffer.create_structured_buffer(
            slang_device, module.Shape, shapes),
        'num_shapes': len(shapes),
        'spheres': buffer.create_structured_buffer(
            slang_device, module.Sphere, spheres),
        'meshes': buffer.create_structured_buffer(
            slang_device, module.TriangleMesh, mesh_descs),
        'vertices': buffer.structured_typed_buffer_from_numpy(
            slang_device, vertices, 'float3'),
        'vertex_normals': buffer.structured_typed_buffer_from_numpy(
            slang_device, normals, 'float3'),
        'triangles': buffer.structured_typed_buffer_from_numpy(
            slang_device, triangles, 'uint3'),
        'bvh_nodes': buffer.create_structured_buffer(
            slang_device, module.BVHNode, bvh_nodes),
        'lights': buffer.create_structured_buffer(
            slang_device, module.PointLight, lights),
        'num_lights': len(lights),
        'max_depth': max_depth
    }

    return bindings, scene, shapes
