import jax.numpy as jnp


def get_intra_distances(particle_positions, displacement_fn):
    return jnp.array([displacement_fn(particle_positions[0, :], particle_positions[1, :]),
                      displacement_fn(particle_positions[0, :], particle_positions[2, :]),
                      displacement_fn(particle_positions[1, :], particle_positions[2, :])
                      ])


def project_particles_back(current_particle_positions, target_distances, displacement_fn):
    c1 = current_particle_positions[0] + (current_particle_positions[1] - current_particle_positions[0]) * target_distances[0] / displacement_fn(current_particle_positions[0], current_particle_positions[1])
    temp = (current_particle_positions[0] + c1) / 2

    #alpha = (-b + jnp.sqrt(b**2 - 4*a*c)) / (2 * a)
    alpha = 1.
    if alpha < 0:
        alpha = -alpha
    c2 = temp + (current_particle_positions[2] - temp) * alpha
    return current_particle_positions[0], c1, c2


LOWEST_ROW = jnp.array([[0., 0., 0., 1.]])


def get_coordinate_frame(three_points, displacement_fn=None):
    #T = jnp.zeros([4, 4])
    #T[-1, -1] = 1
    #T[:-1, -1] = three_points[0, :]
    v1 = three_points[1] - three_points[0]
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = jnp.cross(three_points[2] - three_points[0], v1)
    v2 = v2 / jnp.linalg.norm(v2)  # get vector orthogonal to v2 and v21 and normalize
    v3 = -jnp.cross(v1, v2)
    v3 = v3 / jnp.linalg.norm(v3)
    T = jnp.concatenate([jnp.column_stack([v1, v2, v3, three_points[0]]), LOWEST_ROW], axis=0)
    return T
