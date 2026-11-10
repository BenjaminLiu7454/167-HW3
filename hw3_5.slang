
import hw3_common;

bool intersect_sphere(Sphere s, float3 o, float3 d,                             //HELPERS FROM 3_3
                      float tmin, float tmax, out float t_hit) {
    t_hit = tmax;
    float3 oc = o - s.center;
    float A = dot(d, d);
    float B = 2.0 * dot(d, oc);
    float C = dot(oc, oc) - s.radius * s.radius;

    float2 roots;
    if (!solve_quadratic(A, B, C, roots)) {
        return false;
    }
    float t0 = min(roots.x, roots.y);
    float t1 = max(roots.x, roots.y);

    if (t0 > tmin && t0 < tmax) { t_hit = t0; return true; }
    if (t1 > tmin && t1 < tmax) { t_hit = t1; return true; }
    return false;
}


bool intersect_mesh(
    Shape shape,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    float3 o,
    float3 d,
    float tmin,
    float tmax,
    out float t_hit,
    out float3 barycentrics,
    out uint3 hit_vertices
) {
    t_hit = tmax;
    barycentrics = float3(0.0);
    hit_vertices = uint3(0, 0, 0);

    TriangleMesh mesh = meshes[shape.index];

    // Transform ray into object space.
    // Do NOT normalize d_obj, so t remains unchanged.
    float3 o_obj = xform_pt(shape.world_to_obj, o);
    float3 d_obj = xform_vec(shape.world_to_obj, d);

    bool hit = false;

    // Check every triangle belonging to this mesh.
    for (uint j = mesh.triangle_begin;
         j < mesh.triangle_end;
         j++) {

        // Offset indices into the flattened vertex array.
        uint3 tri = triangles[j] + mesh.vertex_offset;

        float3 p0 = vertices[tri.x];
        float3 p1 = vertices[tri.y];
        float3 p2 = vertices[tri.z];

        float3 b;
        float t;

        if (intersect_tri(p0, p1, p2, o_obj, d_obj, tmin, t_hit, b, t)) {
            hit = true;
            t_hit = t;
            barycentrics = b;
            hit_vertices = tri;
        }
    }

    return hit;
}


bool intersect_shape(
    Shape shape,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    float3 o,
    float3 d,
    float tmin,
    float tmax,
    out float t_hit,
    out float3 barycentrics,
    out uint3 hit_vertices
) {
    t_hit = tmax;
    barycentrics = float3(0.0);
    hit_vertices = uint3(0, 0, 0);

    // Sphere intersection.
    if (shape.type == ShapeType.Sphere) {
        float3 o_obj = xform_pt(shape.world_to_obj, o);
        float3 d_obj = xform_vec(shape.world_to_obj, d);

        return intersect_sphere(spheres[shape.index], o_obj, d_obj, tmin, tmax, t_hit);
    }

    // Triangle mesh intersection.
    return intersect_mesh(shape, meshes, vertices, triangles, o, d, tmin, tmax, t_hit, barycentrics, hit_vertices);
}


float3 hit_normal(
    Shape shape,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<float3> vertex_normals,
    float3 o,
    float3 d,
    float t,
    float3 barycentrics,
    uint3 hit_vertices
) {
    float3 n_obj;

    if (shape.type == ShapeType.Sphere) {

        // Analytic sphere normal.
        Sphere s = spheres[shape.index];

        float3 o_obj = xform_pt(shape.world_to_obj, o);
        float3 d_obj = xform_vec(shape.world_to_obj, d);

        float3 p_obj = o_obj + t * d_obj;

        n_obj = (p_obj - s.center) / s.radius;
    }
    else {

        // Interpolate triangle vertex normals.
        float3 n0 = vertex_normals[hit_vertices.x];
        float3 n1 = vertex_normals[hit_vertices.y];
        float3 n2 = vertex_normals[hit_vertices.z];

        n_obj =
            barycentrics.x * n0 +
            barycentrics.y * n1 +
            barycentrics.z * n2;
    }

    // Transform normal using inverse transpose.
    float3 n_world = xform_vec(
        transpose(shape.world_to_obj),
        n_obj
    );

    return normalize(n_world);
}

bool closest_hit(
    StructuredBuffer<Shape> shapes,
    int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    float3 o,
    float3 d,
    float tmin,
    float tmax,
    out float t_hit,
    out int hit_id,
    out float3 barycentrics,
    out uint3 hit_vertices
) {
    t_hit = tmax;
    hit_id = -1;
    barycentrics = float3(0.0);
    hit_vertices = uint3(0, 0, 0);

    for (int i = 0; i < num_shapes; i++) {
        float t;
        float3 b;
        uint3 tri_vertices;

        if (intersect_shape(shapes[i], spheres, meshes, vertices, triangles, o, d, tmin, t_hit, t, b, tri_vertices)) {
            t_hit = t;
            hit_id = i;
            barycentrics = b;
            hit_vertices = tri_vertices;
        }
    }

    return hit_id >= 0;
}

bool occluded(
    StructuredBuffer<Shape> shapes,
    int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    float3 o,
    float3 d,
    float tmin,
    float tmax
) {
    for (int i = 0; i < num_shapes; i++) {
        float t;
        float3 b;
        uint3 tri_vertices;

        if (intersect_shape(shapes[i], spheres, meshes, vertices, triangles, o, d, tmin, tmax, t, b, tri_vertices)) {
            return true;
        }
    }

    return false;
}

float3 direct_light(
    StructuredBuffer<Shape> shapes,
    int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<PointLight> lights,
    int num_lights,
    float3 p,
    float3 n,
    float3 Kd
) {
    float3 color = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < num_lights; i++) {
        PointLight light = lights[i];

        float3 to_light = light.position - p;
        float dist2 = dot(to_light, to_light);
        float dist = sqrt(dist2);
        float3 l = to_light / dist;

        float cos_theta = dot(n, l);

        if (cos_theta <= 0.0) {
            continue;
        }

        // Check shadows against both spheres and triangle meshes.
        if (occluded(shapes, num_shapes, spheres, meshes, vertices, triangles, p, l, EPSILON, dist - EPSILON)) {
            continue;
        }

        color += Kd * light.intensity * cos_theta / dist2;
    }

    return color;
}




[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,

    uniform float3 background,
    uniform Camera camera,

    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,

    StructuredBuffer<Sphere> spheres,

    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<float3> vertex_normals,
    StructuredBuffer<uint3> triangles,

    StructuredBuffer<PointLight> lights,
    uniform int num_lights,

    uniform int max_depth,

    RWTexture2D<float4> image
) {
    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Camera Ray From 3.4

    float aspect = float(width) / float(height);

    float plane_w = 2.0 * tan(radians(camera.fov_x) / 2.0);
    float plane_h = plane_w / aspect;

    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);

    float3 dir_cam = float3(
        (2.0 * u - 1.0) * plane_w / 2.0,
        (1.0 - 2.0 * v) * plane_h / 2.0,
        -1.0
    );

    float3 ray_org = xform_pt(
        camera.cam_to_world,
        float3(0.0, 0.0, 0.0)
    );

    float3 ray_dir = normalize(
        xform_vec(camera.cam_to_world, dir_cam)
    );


    // MIRROR REFLECTIONS FROM 3.4

    float3 color = float3(0.0, 0.0, 0.0);

    float3 throughput = float3(1.0, 1.0, 1.0);

    float tmin = 0.0;

    for (int b = 0; b < max_depth; b++) {

        float t;
        int hit_id;

        float3 barycentrics;
        uint3 hit_vertices;

        if (!closest_hit(shapes, num_shapes, spheres, meshes, vertices, triangles, ray_org, ray_dir, tmin, 1e30, t, hit_id, barycentrics, hit_vertices)) {
            // Ray missed everything.
            color += throughput * background;
            break;
        }

        Shape shape = shapes[hit_id];

        // World-space intersection position.
        float3 p = ray_org + t * ray_dir;

        // World-space normal.
        // Works for both spheres and triangle meshes.
        float3 n = hit_normal(shape, spheres, vertex_normals, ray_org, ray_dir, t, barycentrics, hit_vertices);

        // Direct illumination.
        color += throughput * direct_light(shapes, num_shapes, spheres, meshes, vertices, triangles, lights, num_lights, p, n, shape.Kd);

        // Stop if the shape is not reflective.
        if (all(shape.Kr == float3(0.0, 0.0, 0.0))) {
            break;
        }

        // Update reflection throughput.
        throughput *= shape.Kr;

        // Calculate next reflected ray.
        ray_org = p;
        ray_dir = reflect(ray_dir, n);

        // Avoid self-intersection.
        tmin = EPSILON;
    }

    image[tid] = float4(color, 1.0);
}
