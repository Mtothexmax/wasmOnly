//! CPU-side mesh data and built-in primitives.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Bounding sphere (center, radius) in mesh space; used for frustum culling.
    pub fn bounds(&self) -> (Vec3, f32) {
        if self.vertices.is_empty() {
            return (Vec3::ZERO, 0.0);
        }
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in &self.vertices {
            let p = Vec3::from(v.pos);
            min = min.min(p);
            max = max.max(p);
        }
        let c = (min + max) * 0.5;
        let r = self
            .vertices
            .iter()
            .map(|v| Vec3::from(v.pos).distance(c))
            .fold(0.0, f32::max);
        (c, r)
    }

    /// Axis-aligned cube centred at the origin with edge length `size`.
    pub fn cube(size: f32) -> Self {
        let h = size * 0.5;
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let mut m = MeshData::default();
        for (n, u, v) in faces {
            let (n, u, v) = (Vec3::from(n), Vec3::from(u), Vec3::from(v));
            let base = m.vertices.len() as u32;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = (n + u * su + v * sv) * h;
                m.vertices.push(Vertex {
                    pos: p.into(),
                    normal: n.into(),
                    uv: [(su + 1.0) * 0.5, 1.0 - (sv + 1.0) * 0.5],
                });
            }
            m.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        m
    }

    /// UV sphere.
    pub fn sphere(radius: f32, segments: u32, rings: u32) -> Self {
        let mut m = MeshData::default();
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let theta = v * std::f32::consts::PI;
            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                let phi = u * std::f32::consts::TAU;
                let n = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                m.vertices.push(Vertex {
                    pos: (n * radius).into(),
                    normal: n.into(),
                    uv: [u, v],
                });
            }
        }
        let stride = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let a = r * stride + s;
                let b = a + stride;
                m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        m
    }

    /// Horizontal plane (XZ) facing +Y.
    pub fn plane(size: f32, uv_scale: f32) -> Self {
        let h = size * 0.5;
        let n = [0.0, 1.0, 0.0];
        let vertices = vec![
            Vertex {
                pos: [-h, 0.0, -h],
                normal: n,
                uv: [0.0, 0.0],
            },
            Vertex {
                pos: [h, 0.0, -h],
                normal: n,
                uv: [uv_scale, 0.0],
            },
            Vertex {
                pos: [h, 0.0, h],
                normal: n,
                uv: [uv_scale, uv_scale],
            },
            Vertex {
                pos: [-h, 0.0, h],
                normal: n,
                uv: [0.0, uv_scale],
            },
        ];
        MeshData {
            vertices,
            indices: vec![0, 2, 1, 0, 3, 2],
        }
    }

    /// 1x1 quad in the XY plane facing +Z; UV (0,0) = top-left (screen/panel convention).
    pub fn quad() -> Self {
        let n = [0.0, 0.0, 1.0];
        let vertices = vec![
            Vertex {
                pos: [-0.5, 0.5, 0.0],
                normal: n,
                uv: [0.0, 0.0],
            },
            Vertex {
                pos: [0.5, 0.5, 0.0],
                normal: n,
                uv: [1.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5, 0.0],
                normal: n,
                uv: [1.0, 1.0],
            },
            Vertex {
                pos: [-0.5, -0.5, 0.0],
                normal: n,
                uv: [0.0, 1.0],
            },
        ];
        MeshData {
            vertices,
            indices: vec![0, 3, 2, 0, 2, 1],
        }
    }

    /// Automatic LOD: vertex clustering on a `cells`³ grid over the bounding box. Vertices in the
    /// same cell *and* with a similar normal (6 axis buckets) merge into their average, so hard
    /// edges survive; collapsed and duplicate triangles are dropped. Fast, deterministic, crude.
    pub fn simplify_cluster(&self, cells: u32) -> MeshData {
        use std::collections::HashMap;
        if self.vertices.is_empty() || cells < 1 {
            return self.clone();
        }
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in &self.vertices {
            min = min.min(Vec3::from(v.pos));
            max = max.max(Vec3::from(v.pos));
        }
        let size = (max - min).max(Vec3::splat(1e-6));
        let c = cells as f32;
        let bucket = |n: Vec3| -> u8 {
            let a = n.abs();
            if a.x >= a.y && a.x >= a.z {
                if n.x >= 0.0 {
                    0
                } else {
                    1
                }
            } else if a.y >= a.z {
                if n.y >= 0.0 {
                    2
                } else {
                    3
                }
            } else if n.z >= 0.0 {
                4
            } else {
                5
            }
        };
        let mut cluster_of: Vec<u32> = Vec::with_capacity(self.vertices.len());
        let mut keys: HashMap<(u32, u32, u32, u8), u32> = HashMap::new();
        let mut acc: Vec<(Vec3, Vec3, [f32; 2], f32)> = vec![];
        for v in &self.vertices {
            let p = Vec3::from(v.pos);
            let q = ((p - min) / size * c).floor().min(Vec3::splat(c - 1.0));
            let key = (
                q.x as u32,
                q.y as u32,
                q.z as u32,
                bucket(Vec3::from(v.normal)),
            );
            let id = *keys.entry(key).or_insert_with(|| {
                acc.push((Vec3::ZERO, Vec3::ZERO, [0.0, 0.0], 0.0));
                acc.len() as u32 - 1
            });
            let a = &mut acc[id as usize];
            a.0 += p;
            a.1 += Vec3::from(v.normal);
            a.2[0] += v.uv[0];
            a.2[1] += v.uv[1];
            a.3 += 1.0;
            cluster_of.push(id);
        }
        let vertices: Vec<Vertex> = acc
            .iter()
            .map(|(p, n, uv, k)| Vertex {
                pos: (*p / *k).into(),
                normal: n.normalize_or(Vec3::Y).into(),
                uv: [uv[0] / k, uv[1] / k],
            })
            .collect();
        let mut seen = std::collections::HashSet::new();
        let mut indices = vec![];
        for t in self.indices.as_chunks::<3>().0 {
            let (a, b, cc) = (
                cluster_of[t[0] as usize],
                cluster_of[t[1] as usize],
                cluster_of[t[2] as usize],
            );
            if a == b || b == cc || a == cc {
                continue;
            }
            let mut k = [a, b, cc];
            k.sort_unstable();
            if seen.insert(k) {
                indices.extend([a, b, cc]);
            }
        }
        MeshData { vertices, indices }
    }
}
