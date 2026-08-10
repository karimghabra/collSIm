#version 460
in vec2 qc;
in vec3 vCenter;
flat in float vRad;
flat in vec3 vColor;
uniform mat4 proj;
uniform vec3 fogColor;
uniform float fogDensity;
out vec4 fragColor;

void main() {
    float r2 = dot(qc, qc);
    if (r2 > 1.0) discard;
    float z = sqrt(1.0 - r2);
    vec3 n = vec3(qc, z);
    vec3 vpos = vCenter + n * vRad;
    vec4 clip = proj * vec4(vpos, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;

    vec3 L = normalize(vec3(0.35, 0.5, 0.8));
    float diff = max(dot(n, L), 0.0);
    float amb = 0.32;
    vec3 V = normalize(-vpos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(n, H), 0.0), 48.0) * 0.35;
    vec3 col = vColor * (amb + 0.75 * diff) + vec3(spec);

    float fog = 1.0 - exp(-fogDensity * max(-vpos.z, 0.0));
    fragColor = vec4(mix(col, fogColor, fog), 1.0);
}
