use crate::banach;

pub struct Boid<E: banach::Element> {
    position: E,
    velocity: E,
}

impl<E: banach::Element> Boid<E> {
    pub fn new(position: E, velocity: E) -> Self {
        Self { position, velocity }
    }
}

#[derive(Clone, Copy)]
pub struct Config<E: banach::Element, S: banach::Space<Element = E>> {
    cohesion_weight: E::ScalarElement,
    alignment_weight: E::ScalarElement,
    separation_weight: E::ScalarElement,
    neighbor_radius: E::NormElement,
    separation_radius: E::NormElement,
    space: S,
}

pub fn next<E: banach::Element, S: banach::Space<Element = E>>(
    boids: &[Boid<E>],
    config: Config<E, S>,
) -> Vec<Boid<E>> {
    let mut new_boids: Vec<Boid<E>> = Vec::new();
    for boid in boids {
        let neighbors: Vec<&Boid<E>> = boids
            .iter()
            .filter(|other| other.position.distance_ref(&boid.position) < config.neighbor_radius)
            .collect();

        let velocity = if neighbors.is_empty() {
            boid.velocity.clone()
        } else {
            let cohesion_velocity =
                cohesion(boid, &neighbors, &config).mul_ref(&config.cohesion_weight);
            let alignment_velocity =
                alignment(boid, &neighbors, &config).mul_ref(&config.alignment_weight);
            let separation_velocity =
                separation(boid, &neighbors, &config).mul_ref(&config.separation_weight);

            boid.velocity
                .add(cohesion_velocity)
                .add(alignment_velocity)
                .add(separation_velocity)
        };

        let position = boid.position.add_ref(&velocity);

        new_boids.push(Boid::new(position, velocity));
    }

    new_boids
}

fn cohesion<E: banach::Element, S: banach::Space<Element = E>>(
    boid: &Boid<E>,
    neighbors: &[&Boid<E>],
    config: &Config<E, S>,
) -> E {
    let neighbor_positions = neighbors.iter().map(|neighbor| neighbor.position.clone());
    let center = config.space.mean(neighbor_positions);
    center.sub(boid.position.clone())
}

fn alignment<E: banach::Element, S: banach::Space<Element = E>>(
    boid: &Boid<E>,
    neighbors: &[&Boid<E>],
    config: &Config<E, S>,
) -> E {
    let neighbor_velocities = neighbors.iter().map(|neighbor| neighbor.velocity.clone());
    let average_velocity = config.space.mean(neighbor_velocities);
    average_velocity.sub_ref(&boid.velocity)
}

fn separation<E: banach::Element, S: banach::Space<Element = E>>(
    boid: &Boid<E>,
    neighbors: &[&Boid<E>],
    config: &Config<E, S>,
) -> E {
    let mut steer_vector = config.space.zero();
    let close_neighbors = neighbors.iter().filter(|neighbor| {
        neighbor.position.distance_ref(&boid.position) < config.separation_radius
    });

    for close_neighbor in close_neighbors {
        steer_vector = steer_vector.add(boid.position.sub_ref(&close_neighbor.position));
    }

    steer_vector
}
