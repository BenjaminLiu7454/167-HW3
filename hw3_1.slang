import hw3_common;

bool intersect_sphere(Sphere s, float3 o, float3 d, float tmin, out float t_hit) {
    t_hit = 0.0;
    float3 oc = o - s.center;
    float A = dot(d, d);
    float B = 2.0 * dot(d, oc);
    float C = dot(oc, oc) - s.radius * s.radius;

    float2 roots;
    if (!solve_quadratic(A, B, C, roots)) {
        return false; // no real roots: ray misses the sphere
    }
    float t0 = roots.x;
    float t1 = roots.y;
    if (t0 > t1) { // sort, in case the solver doesn't
        float tmp = t0; t0 = t1; t1 = tmp;
    }

    // prefer the closer root; fall back to the far one
    // (happens when the ray starts inside the sphere)
    if (t0 > tmin) { t_hit = t0; return true; }
    if (t1 > tmin) { t_hit = t1; return true; }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    
    // --- Camera ray through the pixel center (Fig. 1) ---
    // image plane at z = -1, spans x in [-a, a], y in [-1, 1]; row 0 is the top.
    float aspect = float(width) / float(height);
    float u = (float(tid.x) + 0.5) / float(width);   // 0..1 left -> right
    float v = (float(tid.y) + 0.5) / float(height);  // 0..1 top -> bottom
    float3 ray_org = float3(0.0, 0.0, 0.0);
    float3 ray_dir = float3(aspect * (2.0 * u - 1.0),
                            1.0 - 2.0 * v,
                            -1.0);

    // --- Find the closest hit among all spheres ---
    float closest_t = 1e30;
    int hit_id = -1;
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(spheres[i], ray_org, ray_dir, 0.0, t)) {
            if (t < closest_t) {
                closest_t = t;
                hit_id = i;
            }
        }
    }

    // --- Shade ---
    float3 color = background;
    if (hit_id >= 0) {
        Sphere s = spheres[hit_id];
        float3 p = ray_org + closest_t * ray_dir;
        float3 n = normalize(p - s.center);
        color = (n + 1.0) / 2.0; // map [-1,1] -> [0,1]
    }

    image[tid] = float4(color, 1.0);
}