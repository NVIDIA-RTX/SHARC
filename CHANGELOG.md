## 1.8.0

- Added optional **SHARC_ENABLE_SH_ENCODING** (default: off), which adds directionality support for more accurate cached radiance reconstruction. This is typically needed to avoid extra light leaking from bright specular highlights into unrelated viewing directions.
- When **SHARC_ENABLE_SH_ENCODING** is enabled, both accumulation and resolved buffer storage layouts change to store the extra directional radiance data. Integrations must allocate 32-byte accumulation entries and 24-byte resolved entries instead of the default 16-byte entries.
- Added AGENTS.md for agent-based integration support

## 1.7.2

- Added an experimental responsive lighting mode to support short-lived, high-intensity light sources, controlled by the **SHARC_ENABLE_RESPONSIVE_LIGHTING** define. ```SharcUpdateHit()``` and ```SharcUpdateMiss()``` now include an additional parameter to indicate a responsive signal. This parameter is ignored when responsive lighting is disabled. ```SharcGetCachedRadiance()``` has also been extended with an extra parameter that allows skipping responsive lighting contributions when they are known to have no impact on the current scene
- Fixed a bug where the last vertex contribution could be missed during the update pass
- The cache update pass now uses only 2 entries by default with resampling, which can improve performance and reduce register pressure in some cases
- Added a compact version of the hash grid with 32-bit hash keys
- Redesigned the global hash grid API to support multiple hash grid variants through a unified interface. Each grid must define custom **HASH_GRID_PREFIX** and **HASH_GRID_CONST_PREFIX** values
- Updated documentation

## 1.6.5

- Added optional **SHARC_ENABLE_FADE_ACCELERATION** (default: off). When enabled, each resolve entry tracks whether the current frame's luminance is below the previous frame's. If fading is detected for all 32 tracked frames, the history is reset to match the current frame's sample count, accelerating convergence. Requires ```SharcResolveParameters::frameIndex``` to be set.
- **SHARC_LINEAR_PROBE_WINDOW_SIZE** default increased from 4 to 8.
- Introduced auxiliary function ```SharcLuma()```, shared by the anti-firefly filter and fade acceleration features.
- Refactor:
  - Replaced ```SharcResolveParameters::enableAntiFireflyFilter``` with ```frameIndex``` (required for **SHARC_ENABLE_FADE_ACCELERATION**).
  - Renamed ```SharcPackedData::luminanceM2``` to ```sampleDataExt``` in SharcTypes.h. Carried out the counterpart changes for ```SharcVoxelData::luminanceM2``` accordingly.

## 1.6.3

- SharcTypes.h now defines the base data types needed for correct variable declarations when those types are required before including SharcCommon.h
- Resolved radiance is now stored as 16-bit floats per component. Accumulated/resolved frame ranges are also 16 bits per component. Persistent data still uses a single structured buffer with a 16-byte stride
- Accumulation now uses an exponential weighting scheme, leading to more aggressive history rejection. You may need to increase the accumulated frame count to match the previous behavior
- Performs a linear probe within a bounded window to recover previous data when an element is reallocated in the current frame. The search range is controlled by SHARC_LINEAR_PROBE_WINDOW_SIZE
- Updates to improve GLSL compatibility

## 1.6.0

- Radiance data is now now clearly separated between 'Update' and 'Resolve' passes. SharcParameters::accumulationBuffer(former voxelDataBuffer) is written in the 'Update' pass, while 'SharcParameters::resolvedBuffer'(former voxelDataBufferPrev) is persistent storage populated in the 'Resolve' pass. Both buffers continue to use a 16-byte struct stride
- 'SharcParameters::accumulationBuffer' no longer needs explicit clearing every frame, clear it once before first use
- 'SharcParameters::resolvedBuffer' now stores now stores radiance at full 32-bit precision. Data is still interpreted as uint3 during buffer reads/writes
- 'SHARC_RADIANCE_SCALE' is replaced by SharcParameters::radianceScale. For compatibility the effective scale is the max of the two. 'SharcParameters::radianceScale' improves utilization of the 32-bit per-frame accumulation range, does not affect the persistent data produced by the 'Resolve' pass, and can be adjusted per frame (e.g., with exposure)
- Updated documentation

## 1.5.1

- Deprecated compaction pass option. In small-cache configurations it can degrade performance, especially under heavy motion
- Added optional material demodulation to preserve texture detail. Controlled via the SHARC_MATERIAL_DEMODULATION define
- Updated documentation