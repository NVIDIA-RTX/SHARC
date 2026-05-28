# SHaRC Integration Guide

SHaRC algorithm integration doesn't require substantial modifications to the existing path tracer code. The core algorithm consists of three passes: Update, Resolve, and Render/Query. The first pass uses sparse tracing to populate the world-space radiance cache using the existing path tracer. The second pass resolves and combines newly accumulated data with data from previous frames. The final pass traces rays while querying the cache on hits to accelerate rendering through early termination.

<table>
  <tr>
    <td align="center">
      <img src="images/sample_normal.jpg" alt="Normal" width="49%"/>
      <img src="images/sample_sharc.jpg" alt="SHaRC" width="49%"/><br>
      <em>Image 1. Path traced output at 1 path per pixel (left) and with SHaRC cache usage (right)</em>
    </td>
  </tr>
</table>

## Integration Steps

An implementation of SHaRC using the RTXGI SDK needs to perform the following steps:

At Load-Time

Create the resources:
* `Hash entries` buffer - structured buffer with 8-byte entries that store the hashes
* `Accumulation` buffer - structured buffer with 16-byte entries that store accumulated radiance and sample counts per frame
* `Resolved` buffer - structured buffer with 16-byte entries holding cross-frame accumulated radiance, total samples, and some extra data used in 'Resolve' pass

All buffers should contain the same number of entries, representing the number of scene voxels used for radiance caching. A solid baseline for most scenes can be the usage of $2^{22}$ elements. It is recommended to use power-of-two values. A higher element count is recommended for scenes with high depth complexity. A lower element count reduces memory pressure but increases the risk of hash collisions.

> ⚠️ **Warning:** **All buffers should be initially cleared with '0'**

At Render-Time

* **Populate cache data** using sparse tracing against the scene
* **Combine new cache data with data accumulated from previous frames**
* **Perform tracing** with early path termination using cached data

## Hash Grid Visualization

`Hash grid` visualization itself doesn’t require any GPU resources to be used. The simplest debug visualization uses world space position derived from the primary ray hit intersection.

```C++
HashGridParameters gridParameters;
gridParameters.cameraPosition = g_Constants.cameraPosition;
gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
gridParameters.sceneScale = g_Constants.sharcSceneScale;
gridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

float3 color = HashGridDebugColoredHash(positionWorld, gridParameters);
```

<table>
  <tr>
    <td align="center">
      <img src="images/00_normal.jpg" alt="Normal" width="49%"/>
      <img src="images/00_debug.jpg" alt="Debug" width="49%"/><br>
      <em>Image 2. SHaRC hash grid visualization</em>
    </td>
  </tr>
</table>

The logarithm base controls the distribution of detail levels and the ratio of voxel sizes between neighboring levels. It does not affect the average voxel size. To control voxel size use ```sceneScale``` parameter instead. HashGridParameters::levelBias controls level selection near the camera. It can be used either to add more near-camera levels (finer detail) or to clamp the minimum voxel size to avoid overly fine levels.

## Implementation Details

### Shader Compiler Requirements

SHaRC requires native fp16 shader type support. `SharcTypes.h` stores resolved cache data as `float16_t4`, and enabling `SHARC_USE_FP16` also stores update sample weights as `float16_t3`. For HLSL, compile every shader permutation that includes `SharcTypes.h` or `SharcCommon.h` with DXC `-enable-16bit-types`. This applies to SHaRC Update, SHaRC Resolve, SHaRC Render/Query, and any debug or visualization shaders that include the SHaRC headers.

The shader target and runtime device must expose native 16-bit type support. DXIL targets should use Shader Model 6.2 or newer. For Vulkan/SPIR-V builds driven by DXC, pass `-enable-16bit-types` together with the existing SPIR-V arguments and enable the corresponding 16-bit arithmetic/storage capabilities in the renderer. For GLSL integrations, include `SharcGlslHelpers.h` before `SharcCommon.h` and enable the required fp16 extensions listed there.

### Render Loop Change

Instead of the original trace call, we should have the following three passes with SHaRC:

* SHaRC Update - RT call which updates the cache with the new data on each frame. Requires `SHARC_UPDATE 1` shader define
* SHaRC Resolve - Compute call which combines new cache data with data obtained on the previous frame
* SHaRC Render/Query - RT call which traces scene paths and performs early termination using cached data. Requires `SHARC_QUERY 1` shader define

### Resource Binding

The SDK provides shader-side headers and code snippets that implement most of the steps above. Shader code should include [SharcCommon.h](../Shaders/Include/SharcCommon.h), which already includes [HashGridCommon.h](../Shaders/Include/HashGridCommon.h).

SHaRC uses three main UAV buffers: hash entries, accumulation, and resolved data. These buffers are bound as UAVs in all passes, with different read/write usage depending on the pass (see table below).

Each pass requires UAV barriers to ensure that writes from the previous pass are visible before subsequent passes read or write the cache.

| **Render Pass**  | **Hash Entries** | **Accumulation** | **Resolved** | **Lock Buffer***|
|:-----------------|:----------------:|:----------------:|:------------:|:---------------:|
| SHaRC Update     |        RW        |       Write      |     Read     |       RW        |
| SHaRC Resolve    |        RW        |        RW        |      RW      |                 |
| SHaRC Render     |       Read       |                  |     Read     |                 |

*Buffer is used if SHARC_ENABLE_64_BIT_ATOMICS is set to 0

### SHaRC Update

> ⚠️ **Warning:** Requires `SHARC_UPDATE 1` shader define

This pass runs a modified path tracer for a subset of screen pixels to populate the radiance cache. To reduce cost, only a fraction of pixels should be processed each frame (e.g., one random pixel per 5×5 block, ~4% of paths), with coverage distributed over time.

Each path segment (bounce) in the update pass is treated independently. For every new sample (path), call SharcInit().

On a miss event, call SharcUpdateMiss() and terminate the path.
On a hit, evaluate radiance at the hit point and call SharcUpdateHit().
If SharcUpdateHit() returns false, the path can be terminated early.

After selecting a new ray direction, compute the segment throughput and pass it to SharcSetThroughput(). Once submitted, path throughput can be reset to 1.0, since each segment is accumulated independently. Accumulated radiance should also be reset per segment.

Positions should vary between frames to ensure full-screen coverage over time.

<table>
  <tr>
    <td align="center">
      <img src="images/sharc_update.svg" alt="SHaRC Update loop" width="40%"/><br>
      <em>Figure 1. Path tracer loop during SHaRC Update pass</em>
    </td>
  </tr>
</table>

### SHaRC Resolve

`Resolve` pass is performed using compute shader which runs `SharcResolveEntry()` for each element. This combines per-frame accumulation data with previously resolved data, performs temporal accumulation, handles stale entry eviction, and resets accumulation data for the next frame.
> 📝 **Note:** Check [Resource Binding](#resource-binding) section for details on the required resources and their usage for each pass.

`SharcResolveEntry()` takes maximum number of accumulated frames as an input parameter to control the quality and responsiveness of the cached data. Larger values can increase quality but also increase response times. `staleFrameNumMax` parameter is used to control the lifetime of cached elements, it is used to control cache occupancy

> ⚠️ **Warning:** Small `staleFrameNumMax` values can negatively impact performance, `SHARC_STALE_FRAME_NUM_MIN` constant is used to prevent such behavior.

### SHaRC Render

> ⚠️ **Warning:** Requires `SHARC_QUERY 1` shader define.

During rendering with SHaRC cache usage we should try obtaining cached data using `SharcGetCachedRadiance()` on each eligible hit (typically excluding the primary hit). Upon success, the path tracing loop should be immediately terminated.

<table>
  <tr>
    <td align="center">
      <img src="images/sharc_render.svg" alt="SHaRC Render loop" width="40%"/><br>
      <em>Figure 2. Path tracer loop during SHaRC Render pass</em>
    </td>
  </tr>
</table>

To avoid potential rendering artifacts certain aspects should be taken into account. If the path segment length is less than a voxel size(checked using `GetVoxelSize()`) we should continue tracing until the path segment is long enough to be safely usable. Unlike diffuse lobes, specular ones should be treated with care. For the glossy specular lobe, we can estimate its "effective" cone spread and if it exceeds the spatial resolution of the voxel grid, the cache can be used. Cone spread can be estimated as:

$$2.0 * ray.length * sqrt(0.5 * a^2 / (1 - a^2))$$

where `a` is material roughness squared.

## Responsive Lighting

Responsive lighting mode is intended for light sources that require fast temporal response and may exist only for a short duration (e.g., flashlights or rapidly changing lights). In such cases, standard SHaRC accumulation may react too slowly, leading to visible lag in lighting updates.

This mode can be enabled by defining `SHARC_ENABLE_RESPONSIVE_LIGHTING 1`. Responsive lighting shares the same cache as regular lighting entries are stored together rather than in a separate structure. However, if a hit contains a responsive lighting component and is marked as responsive, the entire signal for that entry is treated as responsive and propagated accordingly through the cache.

When responsive lighting is enabled, the Resolve pass no longer clears accumulation buffer entries. Instead, a full resource clear is required to ensure the accumulation buffer is reset to zero before the `Update` pass begins.

Responsive signal processing introduces additional overhead and may reduce the benefits of long-term accumulation. It should only be enabled when the scene contains transient or rapidly changing light sources that require faster adaptation.

## Directional Radiance Encoding

Directional radiance encoding is intended for cases where cached radiance should preserve directionality for more accurate reconstruction. This is typically needed to avoid extra light coming from bright specular highlights being reconstructed in unrelated viewing directions.

This mode can be enabled by defining `SHARC_ENABLE_SH_ENCODING 1`. When enabled, SHaRC stores additional directional radiance data and reconstructs cached radiance for the current query or update direction.

When using directional radiance encoding, provide the radiance direction in `SharcHitData`. The direction should point from the hit location toward the previous path vertex. The directionality weight controls how strongly the cached signal depends on that direction: use `0` for diffuse radiance and values closer to `1` for sharp glossy or specular radiance. `SharcSetRadianceDirectionWeight()` should used to update this weight after a new direction is sampled.

When directional radiance encoding is enabled, both accumulation and resolved buffer storage layouts change to store the extra directional data. Accumulation buffer entries must use a 32-byte stride and resolved buffer entries must use a 24-byte stride instead of the default 16-byte stride.

Directional radiance encoding increases memory usage and accumulation cost. It should only be enabled when directional reconstruction improves quality, typically in scenes with intense specular highlights or sharp glossy paths.

## Parameters Selection and Debugging

By default, `SHARC_PROPAGATION_DEPTH` is set to `2` when cache resampling is enabled. This value can be increased if longer paths are required to reach light sources in the scene.

SHaRC radiance values are internally premultiplied with `SHARC_RADIANCE_SCALE` and accumulated using 32-bit integer representation per component.

SHaRC operates in world space, so `HashGridParameters::sceneScale` is generally independent of screen resolution and does not need to be adjusted with it.

During rendering, adding a debug heatmap of bounce count can help evaluate cache usage efficiency.

<table>
  <tr>
    <td align="center">
      <img src="images/01_cache_off.jpg" alt="SHaRC off" width="49%"/>
      <img src="images/01_cache_on.jpg" alt="SHaRC on" width="49%"/><br>
      <em>
        Image 3. Tracing depth heatmap with SHaRC off (left) and SHaRC on (right)<br>
        (green = 1 indirect bounce, red = 2+ indirect bounces)
      </em>
    </td>
  </tr>
</table>

> 💡 **Tip:** [SharcCommon.h](../Shaders/Include/SharcCommon.h) provides several methods to verify potential overflow in internal data structures. `SharcDebugBitsOccupancySampleNum()` and `SharcDebugBitsOccupancyRadiance()` can be used to verify consistency in the sample count and corresponding radiance values representation.

`HashGridDebugOccupancy()` should be used to validate cache occupancy. With a static camera around 10-20% of elements should be used on average, on fast camera movement the occupancy will go up. Increased occupancy can negatively impact performance, to control that we can increase the element count as well as decrease the threshold for the stale frames to evict outdated elements more aggressively.

<table>
  <tr>
    <td align="center">
      <img src="images/sample_occupancy.jpg" alt="Cache occupancy debug" width="49%"/><br>
      <em>
        Image 4. Debug overlay to visualize cache occupancy through <code>HashGridDebugOccupancy()</code>
      </em>
    </td>
  </tr>
</table>

## Memory Usage

By default, ```Hash entries``` buffer and two ```Voxel data``` buffers totally require 40 (8 + 16 + 16) bytes per voxel. With `SHARC_ENABLE_SH_ENCODING 1`, accumulation and resolved buffer entries use 32 and 24 bytes respectively, increasing this to 64 (8 + 32 + 24) bytes per voxel. For $2^{22}$ cache elements this will require 160 MiBs of video memory with the default layout or 256 MiBs with directional radiance encoding. Total number of elements may vary depending on the voxel size and scene scale. Larger buffer sizes may be needed to reduce potential hash collisions.
