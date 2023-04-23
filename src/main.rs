use sophus_rs::opt::nlls::CostArgTuple;
use sophus_rs::opt::nlls::{
    apply, CostFnArg, CostTerm, CostTermArg, CostTermSignature, ResidualFn,
};
use tuple_list::{tuple_list, tuple_list_type, Tuple};

#[derive(Copy, Clone)]
struct StereoBaCostTermSignature {
    c: nalgebra::Vector2<f64>,
    entity_indices: tuple_list_type!(usize, usize),
}

impl CostTermSignature<2> for StereoBaCostTermSignature {
    type EntityIndexTuple = tuple_list_type!(usize, usize);
    type Constants = nalgebra::Vector2<f64>;

    fn c_ref(&self) -> &Self::Constants {
        &self.c
    }

    fn idx_ref(&self) -> &Self::EntityIndexTuple {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 2] = [3, 6];
}

#[derive(Copy, Clone)]
struct StereoBa<const E1: char, const E2: char> {}

impl<const E1: char, const E2: char> ResidualFn<9, 2> for StereoBa<E1, E2> {
    type Args = tuple_list_type!(
        CostTermArg<nalgebra::Vector6::<f64>,E1>,
        CostTermArg<nalgebra::Vector3::<f64>,E2>
    );

    type Constants = nalgebra::Vector2<f64>;

    fn cost(args: &Self::Args, _constants: &Self::Constants) -> CostTerm<9, 2> {
        let cost = CostTerm::new(Self::Args::get_dims());

        println!("{:?}", args);
        println!("{} {}", E1, E2);

        if E1 == 'v' {
            cost.hessian.block(0);
        }
        if E2 == 'v' {
            cost.hessian.block(1);
        }

        //constants.clone()

        cost
    }
}

fn main() {
    let poses = vec![
        nalgebra::Vector6::repeat(0.0),
        nalgebra::Vector6::repeat(1.0),
    ];

    let points = vec![
        nalgebra::Vector3::repeat(0.0),
        nalgebra::Vector3::repeat(1.0),
    ];

    let families = (CostFnArg::var(&poses), CostFnArg::cond(&points));
    let families2 = (CostFnArg::var(&poses), CostFnArg::var(&points));

    let cost_terms = vec![StereoBaCostTermSignature {
        c: nalgebra::Vector2::repeat(1.0),
        entity_indices: tuple_list!(1, 0),
    }];

    //  (0, 0), tuple_list!(1, 0), tuple_list!(2, 0)];

    apply(StereoBa {}, &cost_terms, &families.into_tuple_list());
    apply(StereoBa {}, &cost_terms, &families2.into_tuple_list());

    // let h2 = hlist![ Vec{v:42f32}, true, "hello" ];

    // foo(h2);
}
