import torch

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

import numpy as np

def quat_rotate_numpy(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).reshape(-1, 1)
    b = np.cross(q_vec, v, axis=-1) * q_w.reshape(-1, 1) * 2.0
    c = q_vec * np.matmul(q_vec.reshape(shape[0], 1, 3), v.reshape(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_rotate_inverse_numpy(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).reshape(-1, 1)
    b = np.cross(q_vec, v, axis=-1) * q_w.reshape(-1, 1) * 2.0
    c = q_vec * np.matmul(q_vec.reshape(shape[0], 1, 3), v.reshape(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c  # Fix the sign here


if __name__ == "__main__":
    # Add your code here
    q = torch.randn((10, 4))
    v = torch.randn((10, 3))
    o = quat_rotate(q, v)

    q = q.numpy()
    v = v.numpy()
    o_numpy = quat_rotate_numpy(q, v)
    # tell is the two are close
    print(np.allclose(o_numpy, o.numpy()))

    # test inverse
    q = torch.randn((10, 4))
    v = torch.randn((10, 3))
    o = quat_rotate_inverse(q, v)

    q = q.numpy()
    v = v.numpy()
    o_numpy = quat_rotate_inverse_numpy(q, v)
    # tell is the two are close
    print(np.allclose(o_numpy, o.numpy()))

