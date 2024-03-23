use crate::variables::VarKind;

const fn le_than<const N: usize>(
    c: &[char],
    lhs: [usize; N],
    rhs: [usize; N],
) -> std::cmp::Ordering {
    let mut permutation: [usize; N] = [0; N];

    let mut v_count = 0;
    let mut h = 0;
    loop {
        if h >= N {
            break;
        }
        if c[h] == 'm' {
            permutation[h] = v_count;
            v_count += 1;
        }
        h += 1;
    }

    let mut i = 0;
    loop {
        if i >= N {
            break;
        }
        if c[i] == 'f' {
            permutation[i] = v_count;
            v_count += 1;
        }
        i += 1;
    }
    let mut j = 0;
    loop {
        if j >= N {
            break;
        }
        if c[j] == 'c' {
            permutation[j] = v_count;
            v_count += 1;
        }
        j += 1;
    }

    let mut permuted_lhs: [usize; N] = [0; N];
    let mut permuted_rhs: [usize; N] = [0; N];

    let mut k = 0;
    loop {
        if k >= N {
            break;
        }
        permuted_lhs[permutation[k]] = lhs[k];
        permuted_rhs[permutation[k]] = rhs[k];
        k += 1;
    }

    let mut l = 0;
    loop {
        if l >= N {
            break;
        }
        if permuted_lhs[l] < permuted_rhs[l] {
            return std::cmp::Ordering::Less;
        } else if permuted_lhs[l] > permuted_rhs[l] {
            return std::cmp::Ordering::Greater;
        }
        l += 1;
    }
    std::cmp::Ordering::Equal
}

/// Convert VarKind array to char array for comparison
pub fn c_from_var_kind<const N: usize>(var_kind_array: &[VarKind; N]) -> [char; N] {
    let mut c_array: [char; N] = ['0'; N];

    for i in 0..N {
        c_array[i] = match var_kind_array[i] {
            VarKind::Free => 'f',
            VarKind::Conditioned => 'c',
            VarKind::Marginalized => 'm',
        };
    }

    c_array
}

/// Wrapper around the char array for comparison
///
/// f: free variable
/// c: conditioned variable
/// m: marginalized variable
pub struct CompareIdx<C>
where
    C: AsRef<[char]>,
{
    pub(crate) c: C,
}

impl<C> CompareIdx<C>
where
    C: AsRef<[char]>,
{
    /// Compare two cost argument id tuples
    pub fn le_than<const N: usize>(&self, lhs: [usize; N], rhs: [usize; N]) -> std::cmp::Ordering {
        le_than(self.c.as_ref(), lhs, rhs)
    }

    /// Return true if all non-conditioned variables are equal
    pub fn are_all_non_cond_vars_equal(&self, lhs: &[usize], rhs: &[usize]) -> bool {
        let mut i = 0;
        loop {
            if i >= lhs.len() {
                break;
            }
            if self.c.as_ref()[i] != 'c' && lhs[i] != rhs[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Return true if all marginalized variables are equal
    pub fn are_all_marg_vars_equal(&self, lhs: &[usize], rhs: &[usize]) -> bool {
        let mut i = 0;
        loop {
            if i >= lhs.len() {
                break;
            }
            if self.c.as_ref()[i] == 'm' && lhs[i] != rhs[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

mod test {

    #[test]
    fn test() {
        use super::le_than;
        use crate::cost_args::CompareIdx;

        // Length 2
        const VV: [char; 2] = ['f', 'f'];
        const VC: [char; 2] = ['f', 'c'];
        const CV: [char; 2] = ['c', 'f'];
        const CC: [char; 2] = ['c', 'c'];

        assert_eq!(le_than(&VV, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(le_than(&VV, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(le_than(&VC, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(le_than(&VC, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(le_than(&CV, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(le_than(&CV, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(le_than(&CC, [0, 0], [0, 0]), std::cmp::Ordering::Equal);

        const MV: [char; 2] = ['m', 'f'];
        const VM: [char; 2] = ['f', 'm'];
        const MM: [char; 2] = ['m', 'm'];

        assert_eq!(le_than(&MV, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(le_than(&VM, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(le_than(&MM, [0, 0], [0, 0]), std::cmp::Ordering::Equal);

        const VVV: [char; 3] = ['f', 'f', 'f'];

        assert_eq!(
            le_than(&VVV, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VVV, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VVV, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VVV, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VVV, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&VVV, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CVV: [char; 3] = ['c', 'f', 'f'];

        assert_eq!(
            le_than(&CVV, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVV, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVV, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVV, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVV, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVV, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CVC: [char; 3] = ['c', 'f', 'c'];
        assert_eq!(
            le_than(&CVC, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVC, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVC, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVC, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVC, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVC, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CCV: [char; 3] = ['c', 'c', 'f'];

        assert_eq!(
            le_than(&CCV, [0, 0, 0], [0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            le_than(&CCV, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CCV, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CCV, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CCV, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CCV, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CCV, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CVM: [char; 3] = ['c', 'f', 'm'];
        assert_eq!(
            le_than(&CVM, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVM, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVM, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVM, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVM, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVM, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        // Length 4
        const VVVV: [char; 4] = ['f', 'f', 'f', 'f'];
        const CVVV: [char; 4] = ['c', 'f', 'f', 'f'];
        const CCVV: [char; 4] = ['c', 'c', 'f', 'f'];
        const CVCV: [char; 4] = ['c', 'f', 'c', 'f'];
        const VVCC: [char; 4] = ['f', 'f', 'c', 'c'];
        const CCVC: [char; 4] = ['c', 'c', 'f', 'c'];
        const VCCV: [char; 4] = ['f', 'c', 'c', 'f'];

        assert_eq!(
            le_than(&VVVV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VVVV, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVVV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVVV, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CCVV, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            le_than(&CCVV, [0, 1, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CVCV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CVCV, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&VVCC, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            le_than(&VVCC, [0, 0, 0, 1], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&CCVC, [0, 0, 0, 0], [0, 1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&CCVC, [0, 1, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            le_than(&VCCV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VCCV, [0, 0, 1, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );

        const MVVM: [char; 4] = ['m', 'f', 'f', 'm'];
        const MMVV: [char; 4] = ['m', 'm', 'f', 'f'];
        const VMMV: [char; 4] = ['f', 'm', 'm', 'f'];
        const MMMM: [char; 4] = ['m', 'm', 'm', 'm'];

        assert_eq!(
            le_than(&MVVM, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&MMVV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&VMMV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            le_than(&MMMM, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );

        let c: [char; 3] = ['f', 'f', 'c'];
        let mut l: Vec<[usize; 3]> = vec![
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1],
        ];

        let less = CompareIdx { c };
        l.sort_by(|a, b| less.le_than(*a, *b));
    }
}
