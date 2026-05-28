# AGENTS.md - NVIDIA SHARC Radiance Cache Integration

This file provides repository instructions for agentic coding tools. It is intended to be placed at the repository root for the SHARC integration work. Read this entire file before editing renderer code.

## 1. Mission and target behavior

Integrate NVIDIA SHARC, the Spatially Hashed Radiance Cache, into the existing real-time path tracer so the primary rendering path uses SHARC radiance-cache rendering instead of the original path tracing pass.

The target end state is:

```text
SHARC enabled:
    SHARC Update tracing pass
    resource/UAV barrier
    SHARC Resolve compute pass
    resource/UAV barrier
    SHARC Render/Query tracing pass

SHARC disabled or unsupported:
    original path tracer fallback
```

The SHARC path should become the default renderer when `ENABLE_SHARC` or the project equivalent is enabled. Keep the original path tracer available as a debug/reference fallback until SHARC is validated and buildable.

When a user asks for implementation rather than a plan, complete the integration end-to-end in the current coding session as far as the local repository permits. Do not stop after scaffolding or a phase plan unless blocked by missing SDK files, unsupported renderer architecture, unavailable build tools, or missing project-specific information that prevents safe code changes. When blocked, leave the repository buildable and report exact blockers.

## 2. Authoritative sources and allowed reference links

Use the local SHARC SDK checked into the repository or supplied by the user as the source of truth. Do not blindly copy field names from a different branch or tag.

Use the RTXGI Pathtracer sample as the behavioral reference:

- `Pathtracer.cpp` for host-side pass and resource structure.
- `Pathtracer.hlsl` for shader-side update/query hooks.
- `SharcResolve.hlsl` for the resolve compute pass.

Reference URLs in this document must point only to the sample, not to SHARC SDK repository or SHARC SDK documentation links. Use local SDK files and local SDK documentation for SDK details.

Sample reference links:

- RTXGI SHARC pathtracer sample: https://github.com/NVIDIA-RTX/RTXGI/tree/main/Samples/Pathtracer
- Host-side sample reference: https://github.com/NVIDIA-RTX/RTXGI/blob/main/Samples/Pathtracer/Pathtracer.cpp
- Main shader sample reference: https://github.com/NVIDIA-RTX/RTXGI/blob/main/Samples/Pathtracer/Pathtracer.hlsl
- Resolve shader sample reference: https://github.com/NVIDIA-RTX/RTXGI/blob/main/Samples/Pathtracer/SharcResolve.hlsl

Local SDK files to inspect before coding include:

```text
SharcCommon.h
HashGridCommon.h
SharcTypes.h
HashGridTypes.h or equivalent local header used by the SDK version
local integration documentation, if supplied with the SDK
```

Version warning:

- The SHARC SDK and RTXGI sample may use different field names across versions.
- Examples include `gridParameters` vs `hashGridParameters` and `hashMapData` vs `hashGridData`.
- Always inspect the local `SharcCommon.h`, `HashGridCommon.h`, and `SharcTypes.h` files before writing shader code.
- Compile against the local SDK version. Do not assume that the online sample code matches the local SDK exactly.

## 3. Required agentic working rules

1. Start by inspecting the repository structure and local SHARC SDK version.
2. Do not make destructive changes.
3. Do not delete the original path tracer until SHARC builds, runs, and has a debug/reference fallback.
4. Prefer minimal, idiomatic changes that match the project's existing render-pass, descriptor, shader-compilation, resource-management, naming, and configuration architecture.
5. Preserve existing material, camera, scene, acceleration structure, denoising, accumulation, UI/config, and debug systems unless a change is required for SHARC.
6. Make SHARC optional through a compile-time flag and, when the project has runtime settings, a runtime toggle.
7. Keep D3D12 and Vulkan paths in sync if the project supports both.
8. Do not modify third-party SHARC SDK headers unless a local compatibility shim is strictly required.
9. Do not introduce large new framework dependencies.
10. Do not hard-code local absolute paths.
11. Do not fetch SHARC from the network unless the user explicitly asks for that.
12. Before every substantial patch, check `git status --short` and avoid overwriting unrelated user changes.
13. Work in small logical patches internally, but keep moving through the complete SHARC integration when implementation was requested.
14. Do not stop after scaffolding, a plan, or only one render pass unless blocked by missing SDK files, unsupported renderer architecture, unavailable build tools, or missing project-specific information that prevents safe code changes.
15. Fix compile and shader-compile errors caused by the SHARC changes when local build tools are available.
16. After implementation, run the project's normal configure/build or shader-compile checks if available.
17. If a build cannot be run locally, still run static checks such as file search, formatting checks, shader compile scripts, or CMake generation where available.
18. In the final response, list changed files, explain the new SHARC render flow, list new build flags/runtime toggles/shader defines, report build/test commands run, provide manual validation steps, and call out assumptions and unresolved issues.
19. Keep URLs in this file limited to sample links. Do not add SHARC SDK repository or SDK documentation URLs.

## 4. First action: repository discovery

Before editing, map the target project. Use fast search commands similar to these, adjusted for the host shell and available tools:

```bash
pwd
git status --short
find . -maxdepth 3 -type f \( -iname "*path*trace*" -o -iname "*ray*trace*" -o -iname "*shader*" -o -iname "*.hlsl" -o -iname "*.hlsli" -o -iname "*.slang" -o -iname "*.glsl" -o -iname "CMakeLists.txt" \) | sort
rg -n "PathTrace|PathTracer|Pathtracer|TraceRay|DispatchRays|DispatchRay|RayGen|closesthit|miss|bouncesMax|samplesPerPixel" .
rg -n "BindingSet|Descriptor|DescriptorSet|RootSignature|register\(|space[0-9]|UAV|SRV|StructuredBuffer|RWStructuredBuffer|ByteAddressBuffer" .
rg -n "ShaderTable|Pipeline|Permutation|Define|Macro|CompileShader|DXC|Slang|CMake|add_shader|shader compile" .
rg -n "SHARC|Sharc|HashGrid|NRC|NRD|RTXGI" .
```

Find these project-specific locations:

- build system files
- shader include directories
- shader permutation/define setup
- ray tracing pipeline creation
- shader binding table creation
- render loop or render graph pass scheduling
- output render target writing
- descriptor/register-space definitions
- global/per-frame constant buffer definitions
- existing debug UI, runtime settings, config files, and command-line parsing
- path tracer ray generation shader
- path tracing bounce loop
- material sampling and direct lighting evaluation
- denoiser input generation, if present
- GPU resource creation and clear/barrier helpers

If the local SHARC SDK is not present, stop after discovery and report exactly which include files are missing. Required local files normally include:

```text
SharcCommon.h
HashGridCommon.h
SharcTypes.h
HashGridTypes.h or equivalent local header used by the SDK version
```

## 5. SHARC render-time passes

SHARC integration has three render-time passes.

### 5.1 SHARC Update

- Ray tracing or compute-tracing pass.
- Compiled with `SHARC_UPDATE=1`.
- Traces a sparse subset of paths or consumes sparse root surfaces from an existing visibility pass.
- Writes newly observed radiance into the accumulation cache.

### 5.2 SHARC Resolve

- Compute pass.
- Calls `SharcResolveEntry()` once per cache entry.
- Combines per-frame accumulation with previous resolved cache data.
- Handles temporal accumulation and stale entry eviction.
- Resets or manages accumulation data according to the SDK version and responsive-lighting mode.

### 5.3 SHARC Render/Query

- Ray tracing or compute-tracing pass.
- Compiled with `SHARC_QUERY=1`.
- Traces normally but queries resolved cache on eligible indirect-bounce hits.
- On successful cache lookup, adds cached radiance and terminates the path early.

Terminology used throughout this file:

- Primary or visibility rays/hits are the initial camera or visibility rays/hits that establish the root surface for a pixel or path.
- If primary-surface replacement occurs, the replaced surface is still treated as the primary or visibility surface.
- Any path segment launched from a primary/visibility surface, or from a later surface, is an indirect bounce, even if the local shader loop names it bounce 0 or bounce 1.

## 6. Expected high-level host render loop

Adapt this to the engine's render graph or command-list style:

```cpp
if (enableSharc)
{
    EnsureSharcResourcesCreated();
    UpdateSharcConstants(frameConstants);

    if (sharcNeedsClear)
    {
        ClearSharcHashEntries();
        ClearSharcAccumulation();
        ClearSharcResolved();
    }

    // 1. Sparse cache population.
    DispatchSharcUpdateRays(
        DivideRoundUp(renderWidth, sharcDownscaleFactor),
        DivideRoundUp(renderHeight, sharcDownscaleFactor));

    BarrierSharcBuffers();

    // 2. Cache resolve and temporal accumulation.
    DispatchSharcResolveCompute(DivideRoundUp(sharcEntriesNum, 256), 1, 1);

    BarrierSharcBuffers();

    // 3. Final render path using cached radiance.
    DispatchSharcQueryRays(renderWidth, renderHeight);
}
else
{
    DispatchReferencePathTracer(renderWidth, renderHeight);
}
```

If the engine has a render graph, express the same dependencies in graph edges. Do not rely on implicit ordering if the graph requires explicit UAV/resource dependencies.

## 7. Build-system and shader include integration

Add the local SHARC include directory to the shader compiler include path. Prefer the project's existing third-party layout, and do not create a duplicate SDK copy if one already exists.

SHARC requires native fp16 shader type support. For HLSL/DXC shader compilation, add `-enable-16bit-types` to every shader permutation that includes `SharcTypes.h` or `SharcCommon.h`, including Update, Resolve, Query, and debug/visualization shaders. DXIL targets should use Shader Model 6.2 or newer, and the runtime device must support native 16-bit types. For Vulkan/SPIR-V output from DXC, pass `-enable-16bit-types` alongside the existing SPIR-V options and enable the matching device capabilities/extensions.

Use the SDK supplied in the workspace. Do not fetch SHARC from the network unless the user explicitly asked for that.

Add a build option equivalent to:

```cmake
option(ENABLE_SHARC "Enable NVIDIA SHARC radiance cache" ON)
```

If the project uses a different build system, add the equivalent feature flag.

Required shader defines/permutations:

```text
Default/reference path tracer: SHARC_UPDATE=0, SHARC_QUERY=0
SHARC Update permutation:     SHARC_UPDATE=1, SHARC_QUERY=0
SHARC Query permutation:      SHARC_UPDATE=0, SHARC_QUERY=1
SHARC Resolve compute:        SHARC_UPDATE=0, SHARC_QUERY=0
```

## 8. Host-side resources

Create a host-side resource owner, for example:

```cpp
struct SharcResources
{
    uint32_t entriesNum = 1u << 22;
    BufferHandle hashEntries;
    BufferHandle accumulation;
    BufferHandle resolved;
    BindingSetHandle bindingSet;
    BindingLayoutHandle bindingLayout;
};
```

Use the same `entriesNum` for all SHARC buffers.

Recommended initial baseline:

```cpp
entriesNum = 1u << 22; // 4,194,304 entries
```

Make `entriesNum` configurable, preferably as a power of two. Higher values reduce hash collisions for complex scenes. Lower values reduce memory pressure. Accurately determine and validate resource strides for all buffer/resource types using the actual HLSL-visible data layout and type definitions.

### 8.1 Clear/init behavior

At initialization, clear all SHARC buffers to the values expected by the local SDK.

Rules:

1. Read the local `HashGridCommon.h`/`HashGridTypes.h` and the local integration sample to confirm initialization semantics before writing clear code.
2. The local SHARC integration guidance may state that all core buffers should be initially cleared to zero; use zero initialization when the local SDK matches that contract.
3. If the local SDK or local sample exposes a nonzero invalid-key value for hash entries, clear hash entries to that value and document the reason in the code.
4. Clear accumulation and resolved buffers to zero unless the local SDK version explicitly says otherwise.
5. On scene reload, material-demodulation mode change, SHARC parameter reset, or resize that invalidates resources, clear all SHARC cache resources again.
6. For responsive lighting mode, follow the local SDK integration guidance: accumulation may require an explicit full clear before Update because Resolve may not clear it in that mode.

## 9. Descriptor/register binding

Prefer the project's descriptor abstraction. The RTXGI sample binds SHARC resources in a separate descriptor set/register space and uses this HLSL shape:

```hlsl
RWStructuredBuffer<HashGridKey> u_SharcHashEntriesBuffer : register(u0, space3);
RWStructuredBuffer<SharcAccumulationData> u_SharcAccumulationBuffer : register(u2, space3);
RWStructuredBuffer<SharcPackedData> u_SharcResolvedBuffer : register(u3, space3);
```

Adapt register spaces to the target engine. If Vulkan binding macros are used, update both HLSL and Vulkan binding declarations. Keep bindings stable across Update, Resolve, and Query passes.

Bind all SHARC buffers as UAVs for simplicity, but barriers must still make writes visible.

## 10. Constant-buffer additions

Add SHARC fields to the existing global, lighting, or per-frame constant buffer. Respect packing/alignment rules.

Minimum recommended fields:

```cpp
uint32_t sharcEntriesNum;
uint32_t sharcDownscaleFactor;
uint32_t sharcAccumulationFrameNum;
uint32_t sharcStaleFrameNum;

float sharcSceneScale;
float sharcRoughnessMin;
float sharcRadianceScale;
uint32_t sharcDebugMode;

float4 sharcCameraPosition;
float4 sharcCameraPositionPrev;
```

If the project already has camera constants, reuse them when possible, but keep both current and previous SHARC camera positions for Resolve.

Update-pass camera handling:

- If the update pass uses a sparse lower-resolution launch, make sure the generated camera rays still cover the full view over time.
- Store a stable update-pass view if the sample strategy requires it.
- Advance `sharcCameraPositionPrev` only when SHARC update is actually run.

Default tunables to expose:

```text
sharcEntriesNum:              1 << 22 initially, configurable
sharcDownscaleFactor:         5, configurable 1-10
sharcAccumulationFrameNum:    32 initially, configurable 0-100
sharcStaleFrameNum:           64 initially, configurable 0-100
sharcSceneScale:              50.0f initially, configurable 1.0f-100.0f
sharcRoughnessMin:            0.4f initially, configurable 0.0f-1.0f; override material roughness during SharcUpdate to avoid caching local specular highlights
sharcRadianceScale:           start with SDK default or 1e3 if the local SDK comment recommends it
```

Do not hard-code `sharcSceneScale` without debug validation. It controls voxel size in world space and is not tied to screen resolution.

## 11. Pipeline/permutation integration

Add pipeline entries equivalent to:

```cpp
enum class PathTracerPermutation
{
    Reference,
    SharcUpdate,
    SharcQuery,
};
```

Or use the project's existing permutation system.

SHARC Update tracing permutation/pass:

```text
use the existing ray-tracing or compute-tracing mechanism; this may be raygen/closest-hit/miss, inline ray queries, or a project CastRay-style wrapper
compile defines: SHARC_UPDATE=1, SHARC_QUERY=0
output writes: normally not final color except debug/denoiser special handling
```

SHARC Query tracing permutation/pass:

```text
use the same project tracing mechanism as the final path tracer or secondary integrator
compile defines: SHARC_UPDATE=0, SHARC_QUERY=1
output writes: final path-traced output, denoiser inputs, or engine equivalent
```

SHARC Resolve compute PSO:

```text
shader file: SharcResolve.hlsl, SharcResolve.slang, or project equivalent
entry point: sharcResolve or project naming convention
thread group size: 256 x 1 x 1
```

If the engine has shader hot reload or shader reflection, update metadata accordingly.

## 12. Shared shader helper: build SHARC parameters

Avoid copy/pasting SHARC parameter setup in multiple shaders. Prefer a helper function in the path tracer shader or a shared include:

```hlsl
SharcParameters BuildSharcParameters()
{
    SharcParameters p;

    // Use the actual field names from the local SDK.
    p.hashGridParameters.cameraPosition = g_GlobalConstants.sharcCameraPosition.xyz;
    p.hashGridParameters.sceneScale = g_GlobalConstants.sharcSceneScale;
    p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    p.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    p.hashGridData.capacity = g_GlobalConstants.sharcEntriesNum;
    p.hashGridData.hashEntriesBuffer = u_SharcHashEntriesBuffer;
    p.accumulationBuffer = u_SharcAccumulationBuffer;
    p.resolvedBuffer = u_SharcResolvedBuffer;
    p.radianceScale = g_GlobalConstants.sharcRadianceScale;

    return p;
}
```

Important: this is a template, not guaranteed to match every SDK version. If the local SDK does not have `hashGridParameters`, `hashGridData`, `radianceScale`, or the shown buffer fields, adapt to local definitions.

If the local SDK exposes additional fields, initialize them. Do not leave struct fields uninitialized. If the SDK requires radiance scale as a compile-time macro rather than a constant-buffer field, keep that path consistent across Update, Resolve, and Query permutations and document it in code.

## 13. Resolve shader template

Create a resolve shader if none exists. Adapt field names to the local SDK.

```hlsl
[numthreads(256, 1, 1)]
void sharcResolve(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint entryIndex = dispatchThreadId.x;
    if (entryIndex >= g_GlobalConstants.sharcEntriesNum)
        return;

    // Use the actual field names from the local SDK.
    SharcResolveParameters resolveParameters;
    resolveParameters.cameraPositionPrev = g_GlobalConstants.sharcCameraPositionPrev.xyz;
    resolveParameters.accumulationFrameNum = g_GlobalConstants.sharcAccumulationFrameNum;
    resolveParameters.staleFrameNumMax = g_GlobalConstants.sharcStaleFrameNum;
    resolveParameters.frameIndex = g_GlobalConstants.frameIndex;

    SharcResolveEntry(entryIndex, BuildSharcParameters(), resolveParameters);
}
```

Important: this is a template, not guaranteed to match every SDK version.

If the local SDK exposes additional fields, initialize them. Do not leave struct fields uninitialized.

## 14. Path tracing shader integration

Use the existing path tracing loop. Do not rewrite the renderer from scratch.

### 14.1 Required includes and resources

In the main path tracer shader:

```hlsl
#include "SharcCommon.h"
```

Bind the SHARC buffers in the same register space as the host descriptor set.

### 14.2 Primary/visibility rays and indirect bounces

Do not infer primary-vs-indirect behavior only from a local bounce counter. First identify where the renderer creates the primary/visibility surface for the pixel or path.

Common architectures:

1. Monolithic path tracer: the camera ray and bounce loop are in one shader. The first camera hit is the primary/visibility hit; following traced segments are indirect bounces.
2. Split-primary or G-buffer-first renderer: primary/visibility rays are traced before the secondary lighting function. Treat the input surface record as the primary/visibility hit. The first continuation ray launched from it is an indirect bounce and may be eligible for SHARC Query.
3. Secondary-only or no-primary query shader: do not add camera rays just for SHARC Query. Treat the supplied root surface as the primary/visibility hit and query only hits reached by newly traced continuation segments. The Update pass must still obtain root surfaces from sparse camera rays, a visibility/G-buffer pass, or a compacted surface list.
4. Primary-surface replacement, mirror, delta, or transparent visibility chain: treat the replaced surface as the primary/visibility hit. Do not query SHARC at that replaced surface unless the project explicitly approves a primary/visibility-query policy; use SHARC on later indirect bounces by default.

Update may populate cache entries at primary/visibility surfaces and later indirect-bounce surfaces. Query should normally skip primary/visibility surfaces and query only eligible indirect-bounce hits.

### 14.3 Update pass: sparse coverage

The Update pass must not trace every final-resolution pixel each frame unless the user explicitly wants maximum quality at high cost.

Acceptable strategies:

1. Launch at `renderWidth / sharcDownscaleFactor` by `renderHeight / sharcDownscaleFactor`, but map the launch coordinates so rays cover the full camera frustum.
2. Launch full resolution and select one pixel per `N x N` tile using frame-index jitter.
3. Use some noise function or low-discrepancy pattern if the engine already has one.

Do not always trace only the top-left subrectangle. Coverage must be distributed over time. Update should only write to SHARC-related output resources.

### 14.4 Update pass: SHARC minimal roughness usage

During `SHARC_UPDATE`, apply `sharcRoughnessMin` to clamp the material roughness/alpha at each hit point before tracing the next ray.

Do not provide `sharcRoughnessMin` to `SHARC_QUERY` permutations, and do not use it to determine Query pass cache eligibility.

### 14.5 Update pass: per-path initialization

For every new sample/path:

```hlsl
SharcParameters sharcParameters = BuildSharcParameters();
SharcState sharcState;
SharcInit(sharcState);

float3 sampleRadiance = float3(0.0, 0.0, 0.0);
float3 throughput = float3(1.0, 1.0, 1.0);
```

### 14.6 Update pass: miss handling

On a ray miss, evaluate sky/environment as before, then update SHARC:

```hlsl
#if SHARC_UPDATE
    SharcUpdateMiss(sharcParameters, sharcState, skyRadiance);
#endif
```

### 14.7 Hit-data construction

After material and geometry evaluation at a hit:

```hlsl
SharcHitData sharcHitData;
sharcHitData.positionWorld = hitPositionWorld;
sharcHitData.normalWorld = geometryNormalWorld;

#if SHARC_MATERIAL_DEMODULATION
sharcHitData.materialDemodulation = ComputeSharcMaterialDemodulation(material);
#endif

#if SHARC_SEPARATE_EMISSIVE
sharcHitData.emissive = material.emissiveColor;
#endif
```

Use geometry normal in world space unless the project has a proven reason to use a different normal.

### 14.8 Query pass: cache lookup and early termination

In the SHARC Query pass, try to obtain cached radiance at eligible hits reached by indirect bounces. In split-primary or secondary-only shaders, the first hit reached by a continuation ray from the supplied primary/visibility surface is already an indirect-bounce hit.

Do not query the cache on the primary/visibility hit. If surface replacement takes place, the replaced surface is treated as the primary/visibility hit.

Eligibility gate:

```hlsl
#if SHARC_QUERY
// This flag must describe the traced segment, not just a local bounce number.
if (isIndirectBounceHit)
{
    uint gridLevel = HashGridGetLevel(hitPositionWorld, sharcParameters.hashGridParameters);
    float voxelSize = HashGridGetVoxelSize(gridLevel, sharcParameters.hashGridParameters);

    // Use the just-traced indirect segment length, not primary/visibility depth.
    bool validSegmentLength = payload.hitDistance > voxelSize * sqrt(3.0);

    // Use the lobe/roughness at the surface that launched this segment.
    float clampedRoughness = min(originMaterialRoughness, 0.99);
    float alpha = clampedRoughness * clampedRoughness;
    float alpha2 = alpha * alpha;
    float footprint = payload.hitDistance * sqrt(0.5 * alpha2 / max(1.0 - alpha2, 1e-6));
    bool validFootprint = footprint > voxelSize;

    float3 cachedRadiance;
    if (validSegmentLength && validFootprint &&
        SharcGetCachedRadiance(sharcParameters, sharcHitData, cachedRadiance, false))
    {
        sampleRadiance += cachedRadiance * throughput;
        break;
    }
}
#endif
```

Adapt field names and `HashGridGet*` function names to the local SDK. `payload.hitDistance` must be the length of the just-traced indirect segment, not primary camera depth. The `validFootprint` check should use the roughness or lobe width at the surface that launched that segment, not the roughness of the current hit material.

### 14.9 Direct lighting and emissive contribution

Keep the existing direct-lighting evaluation. SHARC should cache the local direct/emissive contribution at hit points during Update, including primary/visibility roots and later indirect-bounce hits. Query normally skips the primary/visibility root even though Update may populate it.

After direct lighting and emissive contribution are evaluated:

```hlsl
#if SHARC_UPDATE
if (!SharcUpdateHit(sharcParameters, sharcState, sharcHitData, sampleRadiance, random))
{
    break;
}
#endif
```

### 14.10 Throughput update

After sampling the next BRDF direction and multiplying by BRDF weight, call:

```hlsl
#if SHARC_UPDATE
SharcSetThroughput(sharcState, throughput);

// SHARC accumulates each segment independently. Reset per-segment state
// following the SDK/sample loop structure.
sampleRadiance = float3(0.0, 0.0, 0.0);
throughput = float3(1.0, 1.0, 1.0);
#endif
```

If the shader structure matches the RTXGI sample, the reset can happen at the beginning of each bounce instead. Keep behavior equivalent: each segment is treated independently after being submitted to SHARC.

### 14.11 Bounce termination

Preserve existing termination logic:

- maximum bounce count
- Russian roulette
- throughput threshold
- material absorption
- denoiser G-buffer capture on primary/visibility hit

Make sure SHARC Query can terminate early after a successful cache lookup.

### 14.12 GLSL compatibility

When compiling SHARC through GLSL/Vulkan paths, include the SDK GLSL compatibility helper before `SharcCommon.h`.

Required order example:

```glsl
#define SHARC_ENABLE_GLSL 1
#include "SharcGlslHelpers.h"
#include "SharcCommon.h"
```

Do not include `SharcCommon.h` directly in GLSL mode unless `SharcGlslHelpers.h` has already been included in the same translation unit.

Make sure GLSL shaders declare/enable the extensions required by `SharcGlslHelpers.h` and `SharcTypes.h`.

For HLSL/D3D12 paths, do not include `SharcGlslHelpers.h`; include `SharcCommon.h` normally.

## 15. Render output behavior

When SHARC is enabled:

- The final color output should come from the SHARC Query pass.
- The Update pass normally should not overwrite the final output, except for explicit debug modes or denoiser-specific data that the project requires.
- Denoiser inputs must still be produced if the project uses NRD or another denoiser.
- Original path tracer should only run when SHARC is disabled, unsupported, or explicitly requested as a debug/reference mode.

## 16. Barriers and synchronization

Add explicit barriers or render-graph dependencies for SHARC resources:

```text
After SHARC Update:
    hash entries UAV barrier
    accumulation UAV barrier
    resolved barrier if read during update and written later

After SHARC Resolve:
    hash entries UAV barrier
    accumulation UAV barrier
    resolved UAV barrier
```

If the engine has automatic UAV barriers, verify that all SHARC passes are represented correctly. Do not disable automatic barriers around SHARC unless you add equivalent explicit barriers. Add clear comments at the render-pass boundaries or render-graph edges explaining that SHARC requires Update writes to be visible to Resolve and Resolve writes to be visible to Query.

## 17. Debug and validation modes

Add at least minimal debug support:

1. SHARC enabled/disabled runtime toggle.
2. SHARC cache reset button or command-line flag.
3. SHARC primary/visibility-hit debug view, where rays are cast from the camera and SHARC radiance is queried at the intersection point for validation only.
4. Reasonable defaults for SHARC element count, scene scale, accumulation frame count, and stale-frame eviction, exposed in the same place as other renderer tuning settings.

Useful SDK helpers may include, depending on version:

```hlsl
HashGridDebugColoredHash(...)
HashGridDebugOccupancy(...)
```

## 18. UI and config

Add runtime controls consistent with the project UI/config style:

```text
enableSharc
sharcResetCache
sharcDownscaleFactor
sharcSceneScale
sharcAccumulationFrameNum
sharcStaleFrameNum
sharcRoughnessMin
sharcDebugMode
sharcMaterialDemodulation
```

Do not expose unsafe or confusing controls unless they help validation.

## 19. Unified end-to-end implementation workflow

Follow this as one continuous implementation pipeline. Do not split the work into a planning-only phase followed by separate delivery phases, and do not stop at an intermediate milestone when the user requested implementation. Discover the repository, make the necessary changes, validate them, and report results in one flow, while still making small logical patches internally.

Begin with repository discovery and immediately map these project-specific targets:

- existing path tracer host file or render module
- existing path tracing ray generation, miss, closest-hit, shared include, and resolve/compute shader files
- render loop or render graph entry points
- GPU buffer creation, clear, upload, and barrier helpers
- descriptor/UAV/SRV binding code, root signatures, descriptor sets, or binding layouts
- shader compilation, permutation, define, reflection, and hot-reload setup
- constant-buffer definitions and per-frame update code
- existing debug UI, config, command-line, and runtime settings
- build-system files that need SHARC include paths and optional `ENABLE_SHARC` plumbing
- local SHARC SDK header layout, including exact field names in `SharcCommon.h`, `HashGridCommon.h`, `SharcTypes.h`, and related local headers

Once the mapping is complete, continue directly into implementation instead of stopping after a plan. Add the local SHARC SDK shader include path, preferring an existing local third-party location such as `External/SHARC` or `External/SHARC/include` when present. Then add the project-equivalent `ENABLE_SHARC` option and runtime setting. Keep the original reference path tracer behind an explicit fallback/debug route and make the normal enabled path mutually exclusive with the reference path unless a deliberate comparison mode runs both.

Create the host-side SHARC resource owner and allocate hash-entry, accumulation, and resolved buffers using the strides and semantics confirmed from the local SDK. Zero-initialize the buffers when the local SDK expects zero clears; otherwise use the SDK-confirmed invalid-key value for hash entries and document it. Reset the cache on first use, scene reload, relevant resize, material-demodulation changes, responsive-lighting mode changes, and explicit user reset.

Add the required descriptors/UAV/SRV bindings once, using the project's existing descriptor abstraction and stable register spaces across SHARC Update, SHARC Resolve, and SHARC Query. Add SHARC fields to the existing global/per-frame constants with correct alignment and initialize every field each frame. Include reasonable defaults for `entriesNum`, `downscaleFactor`, `sceneScale`, accumulation frames, stale-frame eviction, roughness threshold, radiance scale, and debug mode, but expose tuning controls through the project's normal UI/config path where possible.

Add the SHARC Resolve compute shader and pipeline using the local SDK's actual function signatures. Dispatch `ceil(sharcEntriesNum / 256)` thread groups and call `SharcResolveEntry()` once per cache entry. Make sure any local responsive-lighting behavior is honored, including explicit accumulation clears when the SDK requires them.

Add SHARC Update and SHARC Query permutations to the existing tracing pipeline, whether it is DXR/VKRT ray tracing, inline ray queries, or compute shaders using project ray wrappers. Compile Update with `SHARC_UPDATE=1, SHARC_QUERY=0`; compile Query with `SHARC_UPDATE=0, SHARC_QUERY=1`; add DXC `-enable-16bit-types` to all HLSL permutations that include the SHARC headers; keep the reference path with both disabled. Route the enabled render path as:

```text
SHARC Update tracing pass
resource/UAV barrier with comments or render-graph dependencies
SHARC Resolve compute pass
resource/UAV barrier with comments or render-graph dependencies
SHARC Render/Query tracing pass that writes the final output or denoiser inputs
```

Modify the existing path tracing shader rather than replacing the renderer. Add a shared `BuildSharcParameters()` helper, initialize `SharcState` per path/sample, and call the appropriate SHARC functions for miss handling, hit-data construction, update submission, throughput update, and cached-radiance query. Query cached radiance only on eligible indirect-bounce hits by default, with segment-length and roughness/footprint gates adapted to the local SDK and to the just-traced continuation segment. For split-primary or secondary-only shaders, treat the input or replaced surface as the primary/visibility root and the following traced continuation as the first indirect bounce. Preserve existing direct lighting, emissive handling, denoiser/G-buffer writes, maximum-bounce logic, Russian roulette, material absorption, and accumulation behavior.

Add validation and debug support in the same style as the renderer already uses. At minimum provide a SHARC on/off toggle, cache reset control, occupancy or cache visualization when supported, hit/miss or bounce-count debug mode when feasible, and reference fallback for visual comparison. Add clear comments at the render-pass boundaries or render-graph edges explaining why the Update -> Resolve -> Query order and barriers are required.

Run the strongest local verification available before reporting. Prefer the normal project configure/build command and shader compilation step. If the repository has no known build command or the environment lacks required tools, run static checks such as `git status --short`, targeted `rg` searches over changed files, CMake generation when available, shader compile scripts, or formatting checks. Fix compile errors caused by the SHARC integration when local tools are available; only stop with a blocker when missing SDK files, unsupported renderer architecture, unavailable build tools, or missing project-specific information prevents further safe edits.

## 20. Acceptance criteria

The integration is complete only when all applicable criteria are met:

1. Project builds with SHARC disabled.
2. Project builds with SHARC enabled.
3. Local SHARC SDK headers are used; no mismatched branch assumptions remain.
4. SHARC buffers are created with correct stride, capacity, bind flags, and initial state.
5. Hash, accumulation, and resolved buffers are cleared correctly.
6. SHARC descriptor bindings match host and shader register spaces.
7. A SHARC Update tracing permutation/pass exists and compiles with `SHARC_UPDATE=1`.
8. A SHARC Resolve compute pass exists and calls `SharcResolveEntry()` for each cache entry.
9. A SHARC Query tracing permutation/pass exists and compiles with `SHARC_QUERY=1`.
10. All HLSL shader permutations that include SHARC headers compile with DXC `-enable-16bit-types`, and Vulkan/SPIR-V paths enable equivalent native fp16 capabilities.
11. Render loop runs Update, barrier, Resolve, barrier, Query when SHARC is enabled.
12. SHARC is used as an optimization for the original real-time path tracer path.
13. Query pass writes the final output or denoiser input expected by the engine.
14. Update pass uses sparse distributed coverage, not a fixed top-left subrect.
15. Cache query excludes primary/visibility hits, including replaced primary surfaces, or has an explicit project-approved primary/visibility-query policy.
16. Split-primary, secondary-only, and no-primary query shaders classify indirect bounces by traced continuation segments, not only by local bounce index.
17. Cache query checks segment length against voxel size.
18. Specular/glossy paths use a roughness/footprint gate or are excluded from cache query.
19. Denoiser/G-buffer outputs still work if the project uses a denoiser.
20. Scene reload, resource resize, or material-demodulation mode change resets the cache.
21. SHARC Query permutations must compile without referencing `sharcRoughnessMin`; that parameter is only allowed in `SHARC_UPDATE` code.
22. Comments or render-graph labels make the required Update -> Resolve -> Query ordering and barriers clear.
23. Final response lists files changed and commands run.
24. Final response explains the new SHARC render flow and lists new build flags, runtime toggles, and shader defines.

## 21. Final response requirements

The final response after implementation must include:

```text
Summary:
- new SHARC render flow and fallback behavior

Files changed:
- file: what changed

Build flags / runtime toggles / shader defines:
- ...

Validation:
- command: result

Manual validation steps:
- ...

Assumptions:
- ...

Known issues / follow-ups:
- ...
```

## 22. Common mistakes to avoid

- Inferring primary/visibility vs indirect-bounce behavior only from a local bounce number in split-primary or secondary-only shaders.
- Re-tracing primary/visibility rays inside SHARC Query when the renderer already produced primary/visibility surfaces.
- Treating a replaced primary surface as an indirect bounce; replaced surfaces are still primary/visibility surfaces.
- Adding SHARC Query to a no-primary shader but forgetting that SHARC Update still needs sparse root surfaces.
- Creating buffers with wrong strides.
- Forgetting to clear cache resources on first use.
- Clearing accumulation every frame when the local SDK expects Resolve to manage it, unless responsive lighting mode requires it.
- Binding resources to different register spaces in host and shader code.
- Applying denoiser when debug visualization is enabled.
- Running the original path tracer after the SHARC Query pass and overwriting the output.
- Replacing the reference path tracer instead of the real-time path tracing path.
- Dispatching Update at reduced size but only covering the top-left of the camera view.
- Querying the cache on later indirect hits instead of at the first eligible indirect hit.
- Querying cache for short path segments smaller than the voxel size, or using primary/visibility depth instead of the just-traced indirect segment length.
- Querying cache for sharp specular paths without a footprint/roughness gate, or using the hit material roughness instead of the lobe/roughness that launched the segment.
- Forgetting barriers between Update, Resolve, and Query.
- Forgetting Vulkan descriptor set updates when adding D3D12 root bindings, or vice versa.
- Exceeding engine-specific limits, such as root signature binding slots for resources or constants.
- Leaving new constant-buffer fields uninitialized or misaligned.
- Adding a debug UI toggle that changes material demodulation without clearing the cache.
