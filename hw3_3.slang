import hw3_common;

bool intersect_sphere(Sphere s, float3 o, float3 d,
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

// Lambertian shading from all point lights with hard shadows (= HW 3.2)
float3 direct_light(StructuredBuffer<Shape> shapes, int num_shapes,
                    StructuredBuffer<Sphere> spheres,
                    StructuredBuffer<PointLight> lights, int num_lights,
                    float3 p, float3 n, float3 Kd) {
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
        if (occluded(shapes, num_shapes, spheres,
                     p, l, EPSILON, dist - EPSILON)) {
            continue;
        }
        color += Kd * light.intensity * cos_theta / dist2;
    }
    return color;
}

bool closest_hit(StructuredBuffer<Shape> shapes, int num_shapes,
                 StructuredBuffer<Sphere> spheres,
                 float3 o, float3 d, float tmin, float tmax,
                 out float t_hit, out int hit_id) {
    t_hit = tmax;
    hit_id = -1;
    for (int i = 0; i < num_shapes; i++) {
        Shape shape = shapes[i];
        if (shape.type != ShapeType.Sphere) {
            continue;
        }
        float t;
        if (intersect_sphere(spheres[shape.index], o, d, tmin, t_hit, t)) {
            t_hit = t;
            hit_id = i;
        }
    }
    return hit_id >= 0;
}

// Shadow test: is anything between tmin and tmax along the ray?
bool occluded(StructuredBuffer<Shape> shapes, int num_shapes,
              StructuredBuffer<Sphere> spheres,
              float3 o, float3 d, float tmin, float tmax) {
    for (int i = 0; i < num_shapes; i++) {
        Shape shape = shapes[i];
        if (shape.type != ShapeType.Sphere) {
            continue;
        }
        float t;
        if (intersect_sphere(spheres[shape.index], o, d, tmin, tmax, t)) {
            return true;
        }
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // --- Camera ray through the pixel center (same as 3.1) ---
    float aspect = float(width) / float(height);
    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);
    float3 ray_org = float3(0.0, 0.0, 0.0);
    float3 ray_dir = normalize(float3(aspect * (2.0 * u - 1.0), 1.0 - 2.0 * v, -1.0));

    // --- Mirror bounces, written as a loop instead of recursion ---
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0); // product of Kr along the path
    float tmin = 0.0; // camera ray: t > 0; bounce rays: t > EPSILON

    for (int b = 0; b < max_depth; b++) {
        float t;
        int hit_id;
        if (!closest_hit(shapes, num_shapes, spheres,
                         ray_org, ray_dir, tmin, 1e30, t, hit_id)) {
            // missed everything: see the background (through any mirrors so far)
            color += throughput * background;
            break;
        }

        Shape shape = shapes[hit_id];
        Sphere s = spheres[shape.index];
        float3 p = ray_org + t * ray_dir;
        float3 n = normalize(p - s.center);

        // diffuse part of this surface, tinted by the mirrors we bounced off
        color += throughput * direct_light(shapes, num_shapes, spheres,
                                           lights, num_lights, p, n, shape.Kd);

        // not a mirror: the path ends here
        if (all(shape.Kr == float3(0.0, 0.0, 0.0))) {
            break;
        }

        // continue in the mirror direction
        throughput *= shape.Kr;
        ray_org = p;
        ray_dir = reflect(ray_dir, n);
        tmin = EPSILON; // avoid re-hitting the surface we just left
    }

    image[tid] = float4(color, 1.0);
}