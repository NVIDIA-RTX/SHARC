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