#version 460
// Instanced sphere impostors: every heavy atom of every near-LOD molecule.
// The straight atom template (contour s, cross-section x,y) is mapped onto
// the deformed filament via interpolated bead positions + transported frames.
layout(std430, binding = 0) readonly buffer Pos { vec4 pos[]; };
layout(std430, binding = 6) readonly buffer Frames { vec4 frm[]; };
layout(std430, binding = 7) readonly buffer Atoms { vec4 atom[]; };
layout(std430, binding = 8) readonly buffer NearIds { uint nearId[]; };
layout(std430, binding = 1) readonly buffer NearOffs { vec4 nearOff[]; };
layout(std430, binding = 2) readonly buffer Ages { float spawnSec[]; };
uniform float nowSec;

uniform mat4 view, proj;
uniform int natoms, nBeads;
uniform float segLen, radScale, dPeriod;
uniform int colorMode;   // 0 element 1 class 2 chain 3 molecule 4 D-phase

out vec2 qc;
out vec3 vCenter;
flat out float vRad;
flat out vec3 vColor;

vec3 qrot(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }

vec3 hsv(float h, float s, float v) {
    vec3 k = abs(fract(h + vec3(0., 2. / 3., 1. / 3.)) * 6.0 - 3.0) - 1.0;
    return v * mix(vec3(1.0), clamp(k, 0.0, 1.0), s);
}

void main() {
    uint inst = uint(gl_InstanceID);
    uint ai = inst % uint(natoms);
    uint slot = inst / uint(natoms);
    uint mol = nearId[slot];
    vec3 imgOff = nearOff[slot].xyz;
    vec4 a = atom[ai];
    uint pk = floatBitsToUint(a.w);
    uint elem = pk & 3u;
    uint cls = (pk >> 4) & 7u;
    uint chain = (pk >> 8) & 3u;

    float f = a.x / segLen;
    int k = clamp(int(f), 0, nBeads - 2);
    float t = f - float(k);
    uint b0 = mol * uint(nBeads) + uint(k);
    vec3 p = mix(pos[b0].xyz, pos[b0 + 1u].xyz, t);
    vec4 q0 = frm[b0], q1 = frm[b0 + 1u];
    if (dot(q0, q1) < 0.0) q1 = -q1;
    vec4 q = normalize(mix(q0, q1, t));
    p += qrot(q, vec3(a.y, a.z, 0.0)) + imgOff;

    const float rads[4] = float[](0.170, 0.155, 0.152, 0.180);
    float R = rads[elem] * radScale;

    vec3 col;
    if (colorMode == 0) {
        const vec3 cpk[4] = vec3[](vec3(0.55), vec3(0.19, 0.31, 0.97),
                                   vec3(1.0, 0.05, 0.05), vec3(1.0, 1.0, 0.19));
        col = cpk[elem];
    } else if (colorMode == 1) {
        const vec3 cc[5] = vec3[](vec3(0.62), vec3(0.24, 0.65, 0.42),
                                  vec3(0.15, 0.35, 0.85), vec3(0.85, 0.15, 0.15),
                                  vec3(1.0, 0.67, 0.0));
        col = cc[min(cls, 4u)];
    } else if (colorMode == 2) {
        const vec3 ch[3] = vec3[](vec3(0.4, 0.65, 0.95), vec3(0.55, 0.8, 0.55),
                                  vec3(0.95, 0.6, 0.35));
        col = ch[min(chain, 2u)];
    } else if (colorMode == 3) {
        col = hsv(fract(float(mol) * 0.61803399), 0.55, 0.9);
    } else {
        col = hsv(fract(a.x / dPeriod), 0.75, 0.95);
    }

    float glow = clamp(1.0 - (nowSec - spawnSec[mol]) / 5.0, 0.0, 1.0);
    col = mix(col, vec3(1.0, 0.95, 0.35), glow * 0.85);

    vec2 corner = vec2(float((gl_VertexID & 1) * 2 - 1), float((gl_VertexID >> 1) * 2 - 1));
    vec4 vc = view * vec4(p, 1.0);
    vCenter = vc.xyz;
    vc.xy += corner * R;
    gl_Position = proj * vc;
    qc = corner;
    vRad = R;
    vColor = col;
}
