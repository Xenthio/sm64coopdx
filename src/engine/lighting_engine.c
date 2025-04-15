#include "lighting_engine.h"
#include "math_util.h"
#include "surface_collision.h"
#include "pc/gfx/gfx_pc.h"
#include "pc/lua/utils/smlua_math_utils.h"
#include "pc/debuglog.h"
#include "data/dynos_cmap.cpp.h"

#define LE_MAX_LIGHTS 32

static Color sAmbientColor;
static void* sLights = NULL;
static s32 sLightID = 0;

static inline void color_set(Color color, u8 r, u8 g, u8 b) {
    color[0] = r;
    color[1] = g;
    color[2] = b;
}

#ifdef __SSE__
void le_calculate_vertex_lighting(f32 x, f32 y, f32 z, Vtx_t* v, Color out, bool useVertexColors, __m128 mat0, __m128 mat1, __m128 mat2, __m128 mat3) {
#else
void le_calculate_vertex_lighting(f32 x, f32 y, f32 z, Vtx_t* v, Color out, bool useVertexColors, float* mpMatrix) {
#endif
    if (sLights == NULL) { return; }

    f32 r = sAmbientColor[0];
    f32 g = sAmbientColor[1];
    f32 b = sAmbientColor[2];
    if (useVertexColors) {
        r *= (v->cn[0] / 255.0f);
        g *= (v->cn[0] / 255.0f);
        b *= (v->cn[0] / 255.0f);
    }

    f32 weight = 1.0f;
    for (struct LELight* light = hmap_begin(sLights); light != NULL; light = hmap_next(sLights)) {
#ifdef __SSE__
        __m128 pos0 = _mm_set1_ps(light->posX);
        __m128 pos1 = _mm_set1_ps(light->posY);
        __m128 pos2 = _mm_set1_ps(light->posZ);

        __m128 pos = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(pos0, mat0), _mm_mul_ps(pos1, mat1)), _mm_mul_ps(pos2, mat2)), mat3);
        f32 lX = pos[0]; // gfx_adjust_x_for_aspect_ratio(pos[0]);
        f32 lY = pos[1];
        f32 lZ = pos[2];
#else
        f32 lX = x * mpMatrix[0][0] + y * mpMatrix[1][0] + z * mpMatrix[2][0] + mpMatrix[3][0];
        f32 lY = x * mpMatrix[0][1] + y * mpMatrix[1][1] + z * mpMatrix[2][1] + mpMatrix[3][1];
        f32 lZ = x * mpMatrix[0][2] + y * mpMatrix[1][2] + z * mpMatrix[2][2] + mpMatrix[3][2];
        lX = gfx_adjust_x_for_aspect_ratio(lX);
#endif
        
        f32 diffX = lX - x;
        f32 diffY = lY - y;
        f32 diffZ = lZ - z;
        f32 dist = (diffX * diffX) + (diffY * diffY) + (diffZ * diffZ);
        f32 radius = light->radius * light->radius;
        if (dist > radius) { continue; }

        f32 brightness = (1 - (dist / radius)) * light->intensity;
        r += light->colorR * brightness;
        g += light->colorG * brightness;
        b += light->colorB * brightness;
        weight += brightness;
    }

    out[0] = min(r / weight, 255);
    out[1] = min(g / weight, 255);
    out[2] = min(b / weight, 255);
}

void le_calculate_lighting_color(Vec3f pos, Color out, f32 lightIntensityScalar) {
    if (sLights == NULL) { return; }

    f32 r = sAmbientColor[0];
    f32 g = sAmbientColor[1];
    f32 b = sAmbientColor[2];

    f32 weight = 1.0f;
    for (struct LELight* light = hmap_begin(sLights); light != NULL; light = hmap_next(sLights)) {
        f32 diffX = light->posX - pos[0];
        f32 diffY = light->posY - pos[1];
        f32 diffZ = light->posZ - pos[2];
        f32 dist = (diffX * diffX) + (diffY * diffY) + (diffZ * diffZ);
        f32 radius = light->radius * light->radius;
        if (dist > radius) { continue; }

        f32 brightness = (1 - (dist / radius)) * light->intensity * lightIntensityScalar;
        r += light->colorR * brightness;
        g += light->colorG * brightness;
        b += light->colorB * brightness;
        weight += brightness;
    }

    out[0] = min(r / weight, 255);
    out[1] = min(g / weight, 255);
    out[2] = min(b / weight, 255);
}

void le_calculate_lighting_dir(Vec3f pos, Vec3f out) {
    if (sLights == NULL) { return; }

    Vec3f lightingDir = { 0, 0, 0 };
    s32 count = 1;
    for (struct LELight* light = hmap_begin(sLights); light != NULL; light = hmap_next(sLights)) {
        f32 diffX = light->posX - pos[0];
        f32 diffY = light->posY - pos[1];
        f32 diffZ = light->posZ - pos[2];
        f32 dist = (diffX * diffX) + (diffY * diffY) + (diffZ * diffZ);
        f32 radius = light->radius * light->radius;
        if (dist > radius) { continue; }

        Vec3f dir = {
            pos[0] - light->posX,
            pos[1] - light->posY,
            pos[2] - light->posZ,
        };
        vec3f_normalize(dir);

        f32 intensity = (1 - (dist / radius)) * light->intensity;
        lightingDir[0] += dir[0] * intensity;
        lightingDir[1] += dir[1] * intensity;
        lightingDir[2] += dir[2] * intensity;

        count++;
    }

    out[0] = lightingDir[0] / (f32)(count);
    out[1] = lightingDir[1] / (f32)(count);
    out[2] = lightingDir[2] / (f32)(count);
    vec3f_normalize(out);
}

s32 le_add_light(f32 x, f32 y, f32 z, u8 r, u8 g, u8 b, f32 radius, f32 intensity) {
    if (sLights == NULL) {
        sLights = hmap_create(true);
    } else if (hmap_len(sLights) >= LE_MAX_LIGHTS) {
        return 0;
    }

    struct LELight* light = calloc(1, sizeof(struct LELight));
    light->posX = x;
    light->posY = y;
    light->posZ = z;
    light->colorR = r;
    light->colorG = g;
    light->colorB = b;
    light->radius = radius;
    light->intensity = intensity;
    hmap_put(sLights, ++sLightID, light);
    return sLightID;
}

void le_remove_light(s32 id) {
    if (sLights == NULL || id <= 0) { return; }

    free(hmap_get(sLights, id));
    hmap_del(sLights, id);
}

s32 le_get_light_count(void) {
    if (sLights == NULL) { return 0; }
    return hmap_len(sLights);
}

void le_set_ambient_color(u8 r, u8 g, u8 b) {
    color_set(sAmbientColor, r, g, b);
}

void le_set_light_pos(s32 id, f32 x, f32 y, f32 z) {
    if (sLights == NULL || id <= 0) { return; }

    struct LELight* light = hmap_get(sLights, id);
    if (light == NULL) { return; }
    light->posX = x;
    light->posY = y;
    light->posZ = z;
}

void le_set_light_color(s32 id, u8 r, u8 g, u8 b) {
    if (sLights == NULL || id <= 0) { return; }

    struct LELight* light = hmap_get(sLights, id);
    if (light == NULL) { return; }
    light->colorR = r;
    light->colorG = g;
    light->colorB = b;
}

void le_set_light_radius(s32 id, f32 radius) {
    if (sLights == NULL || id <= 0) { return; }

    struct LELight* light = hmap_get(sLights, id);
    if (light == NULL) { return; }
    light->radius = radius;
}

void le_set_light_intensity(s32 id, f32 intensity) {
    if (sLights == NULL || id <= 0) { return; }

    struct LELight* light = hmap_get(sLights, id);
    if (light == NULL) { return; }
    light->intensity = intensity;
}

void le_clear(void) {
    if (sLights == NULL) { return; }

    for (struct LELight* light = hmap_begin(sLights); light != NULL; light = hmap_next(sLights)) {
        free(light);
    }
    hmap_clear(sLights);
    sLightID = 0;
    sAmbientColor[0] = 0;
    sAmbientColor[1] = 0;
    sAmbientColor[2] = 0;
}

void le_shutdown(void) {
    if (sLights == NULL) { return; }

    le_clear();
    hmap_destroy(sLights);
    sLights = NULL;
}
