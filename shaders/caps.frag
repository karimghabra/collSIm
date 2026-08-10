#version 460
in vec2 lp;
in vec3 va_v, vb_v;
flat in vec3 vColor;
flat in float vR;
uniform mat4 proj;
uniform vec3 fogColor;
uniform float fogDensity;
out vec4 fragColor;

void main() {
    float u = clamp(lp.x, 0.0, 1.0);
    float lat = lp.y;
    // round the ends: outside [0,1] the lateral limit shrinks like a sphere
    float overshoot = max(max(-lp.x, lp.x - 1.0), 0.0);
    float axn = overshoot * length(vb_v.xy - va_v.xy) / vR;   // in radius units
    float lim2 = 1.0 - axn * axn;
    if (lim2 <= 0.0 || lat * lat > lim2) discard;

    float nz = sqrt(max(1.0 - lat * lat - axn * axn, 0.0));
    vec3 n = normalize(vec3(0.0, lat, nz));   // approximate view-space normal
    vec3 vpos = mix(va_v, vb_v, u);
    vpos.z += nz * vR;
    vec4 clip = proj * vec4(vpos, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;

    vec3 L = normalize(vec3(0.35, 0.5, 0.8));
    float diff = max(dot(n, L), 0.0);
    vec3 col = vColor * (0.35 + 0.7 * diff);
    float fog = 1.0 - exp(-fogDensity * max(-vpos.z, 0.0));
    fragColor = vec4(mix(col, fogColor, fog), 1.0);
}
