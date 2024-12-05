#[allow(unused_imports)]
pub use itertools as __itertools;
#[allow(unused_imports)]
pub use std::iter as __std_iter;

#[macro_export]
/// Adapted from itertools izip! macro, using zip_eq instead of zip
macro_rules! izip_eq {
    // @closure creates a tuple-flattening closure for .map() call. usage:
    // @closure partial_pattern => partial_tuple , rest , of , iterators
    // eg. izip_eq!( @closure ((a, b), c) => (a, b, c) , dd , ee )
    ( @closure $p:pat => $tup:expr ) => {
        |$p| $tup
    };

    // The "b" identifier is a different identifier on each recursion level thanks to hygiene.
    ( @closure $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )* ) => {
        $crate::izip_eq!(@closure ($p, b) => ( $($tup)*, b ) $( , $tail )*)
    };

    // unary
    ($first:expr $(,)*) => {
        $crate::izip_eq::__std_iter::IntoIterator::into_iter($first)
    };

    // binary
    ($first:expr, $second:expr $(,)*) => {
        {
            use $crate::izip_eq::__itertools::Itertools;
            $crate::izip_eq!($first).zip_eq($second)
        }
    };

    // n-ary where n > 2
    ( $first:expr $( , $rest:expr )* $(,)* ) => {
        {
            use $crate::izip_eq::__itertools::Itertools;
            $crate::izip_eq!($first)
            $(
                .zip_eq($rest)
            )*
            .map(
                $crate::izip_eq!(@closure a => (a) $( , $rest )*)
            )
        }
    };
}
