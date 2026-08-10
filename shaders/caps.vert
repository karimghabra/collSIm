#version 460
// Far-LOD: one flat-shaded capsule impostor per filament segment.
layout(std430, binding = 0) readonly buffer Pos { vec4 pos[]; };
layout(std430, binding = 9) readonly buffer NearMask { uint nearMask[]; };
layout(std430, binding = 21) readonly buffer Ages { float spawnSec[]; };
uniform float nowSec;

uniform mat4 view, proj;
uniform int nBeads;
uniform float tubeR, dPeriod, segLen;
uniform int colorMode;
uniform int nSegTotal;
uniform int imgMode;       // 0 primary only, 1 z-images (3), 2 full 27
uniform vec3 boxSize;

out vec2 lp;           // x: axial param (can exceed [0,1] into caps), y: lateral [-1,1]
out vec3 va_v, vb_v;
flat out vec3 vColor;
flat out float vR;

vec3 hsv(float h, float s, float v) {
    vec3 k = abs(fract(h + vec3(0., 2. / 3., 1. / 3.)) * 6.0 - 3.0) - 1.0;
    return v * mix(vec3(1.0), clamp(k, 0.0, 1.0), s);
}

void main() {
    uint inst = uint(gl_InstanceID);
    uint seg = inst % uint(nSegTotal);
    uint img = inst / uint(nSegTotal);
    vec3 ioff = vec3(0.0);
    bool primary = true;
    if (imgMode == 1) {
        ioff = vec3(0.0, 0.0, float(int(img) - 1)) * boxSize;
        primary = (img == 1u);
    } else if (imgMode == 2) {
        ivec3 o = ivec3(int(img) % 3 - 1, (int(img) / 3) % 3 - 1, int(img) / 9 - 1);
        ioff = vec3(o) * boxSize;
        primary = all(equal(o, ivec3(0)));
    }
    uint mol = seg / uint(nBeads - 1);
    uint k = seg % uint(nBeads - 1);
    if (primary && nearMask[mol] == 1u) {   // primary image drawn as atoms
        gl_Position = vec4(2e9, 2e9, 2e9, 1.0);
        return;
    }
    uint b = mol * uint(nBeads) + k;
    vec3 A = pos[b].xyz + ioff, B = pos[b + 1u].xyz + ioff;
    vec3 av = (view * vec4(A, 1.0)).xyz;
    vec3 bv = (view * vec4(B, 1.0)).xyz;

    if (colorMode == 3) vColor = hsv(fract(float(mol) * 0.61803399), 0.55, 0.9);
    else if (colorMode == 4) vColor = hsv(fract((float(k) + 0.5) * segLen / dPeriod), 0.75, 0.95);
    else vColor = mix(vec3(0.58, 0.62, 0.70), hsv(fract(float(mol) * 0.61803399), 0.25, 0.75), 0.5);
    float glow = clamp(1.0 - (nowSec - spawnSec[mol]) / 5.0, 0.0, 1.0);
    vColor = mix(vColor, vec3(1.0, 0.95, 0.35), glow * 0.85);

    vec2 ab = bv.xy - av.xy;
    float l2d = length(ab);
    vec2 ax = l2d > 1e-4 ? ab / l2d : vec2(1, 0);
    vec2 pp = vec2(-ax.y, ax.x);

    int vid = gl_VertexID;              // 0..3 strip
    float au = float(vid >> 1);         // 0 at A end, 1 at B end
    float lat = float((vid & 1) * 2 - 1);
    float R = tubeR;
    vec2 p2 = mix(av.xy - ax * R, bv.xy + ax * R, au) + pp * (lat * R);
    float zz = mix(av.z, bv.z, au);
    gl_Position = proj * vec4(p2, zz, 1.0);
    float ext = R / max(l2d, 1e-4);
    lp = vec2(mix(-ext, 1.0 + ext, au), lat);
    va_v = av;
    vb_v = bv;
    vR = R;
}
