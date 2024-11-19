use crate::rules_engine::state::Unit;

pub struct FullUnitsIterator<'a> {
    units: &'a [Unit],
    max_units: usize,
    current_unit_id: usize,
    current_index: usize,
}

impl<'a> FullUnitsIterator<'a> {
    pub fn new(units: &'a [Unit], max_units: usize) -> FullUnitsIterator {
        FullUnitsIterator {
            units,
            max_units,
            current_unit_id: 0,
            current_index: 0,
        }
    }
}

impl<'a> Iterator for FullUnitsIterator<'a> {
    type Item = Option<&'a Unit>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_unit_id >= self.max_units {
            return None;
        }

        let result = self
            .units
            .get(self.current_index)
            .filter(|unit| unit.id == self.current_unit_id);
        self.current_unit_id += 1;
        if result.is_some() {
            self.current_index += 1;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rstest::rstest;

    #[rstest]
    #[case(
        vec![],
        vec![None; 4],
    )]
    #[case(
        vec![Unit::with_id(1), Unit::with_id(2)],
        vec![None, Some(Unit::with_id(1)), Some(Unit::with_id(2)), None],
    )]
    #[case(
        vec![Unit::with_id(0), Unit::with_id(3)],
        vec![Some(Unit::with_id(0)), None, None, Some(Unit::with_id(3))],
    )]
    fn test_full_units_iterator(
        #[case] units: Vec<Unit>,
        #[case] expected: Vec<Option<Unit>>,
    ) {
        let iterator = FullUnitsIterator::new(&units, 4);
        let result = iterator.map(|opt_u| opt_u.copied()).collect_vec();
        assert_eq!(result, expected);
    }
}
