use core::{
    ops::Index,
    time::Duration,
};

use log::warn;
use sophus_geo::{
    prelude::*,
    region::interval::Interval,
};

extern crate alloc;

/// has time_stamp method
pub trait HasTimeStamp: Clone {
    /// returns time-stamp
    fn time_stamp(&self) -> f64;
}

/// has interpolate method
pub trait HasInterpolate: HasTimeStamp {
    /// interpolate between self and other with parameter w, where w is typically in [0, 1]
    fn interpolate(&self, other: &Self, w: f64) -> Self;
}

/// Time series
///
/// A time series is a collection of items that are ordered by their time stamp.
/// It offers efficient access to the nearest item to a given time stamp, but also random access
/// to items by index.
///
/// The time series is implemented as a double-ended queue (VecDeque) of items, which is kept sorted
/// by time stamp.
///
/// The following operations are especially efficient:
///
/// - Adding an item with a time stamp that is newer than the last item in the time series. This is
///   the common case when adding items in order.
/// - Accessing the first and last item in the time series.
/// - Access item by index.
/// - Pruning older data by time duration.
/// - Finding the nearest item to a given time stamp.
#[derive(Clone)]
pub struct TimeSeries<T: HasTimeStamp> {
    sorted_data: alloc::collections::vec_deque::VecDeque<T>,
}

/// item with index
pub struct IndexedItem<'a, T> {
    /// the index in the time series
    pub index: usize,
    /// reference to item
    pub item: &'a T,
}

impl<T: HasTimeStamp> Default for TimeSeries<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: HasTimeStamp> TimeSeries<T> {
    /// Create a new empty time series.
    pub fn new() -> Self {
        TimeSeries {
            sorted_data: alloc::collections::vec_deque::VecDeque::new(),
        }
    }

    /// reserve space for n items
    pub fn reserve(&mut self, n: usize) {
        self.sorted_data.reserve(n);
    }

    /// Find the nearest item in the time series to the given time stamp and return a reference to
    /// it.
    pub fn find_nearest(&self, time: f64) -> Option<IndexedItem<T>> {
        self.find_nearest_within(time, f64::INFINITY)
    }

    /// Find the nearest item in the time series to the given time stamp and return a reference to
    /// it.
    pub fn find_nearest_within(&self, time: f64, max_dist: f64) -> Option<IndexedItem<T>> {
        if !time.is_finite() {
            warn!(
                "TimeSeries::find_nearest_within: skipping infinite time stamp {}",
                time
            );
            return None;
        }

        if self.sorted_data.is_empty() {
            return None;
        }

        let index = self
            .sorted_data
            .partition_point(|probe| probe.time_stamp() < time);

        if index == self.sorted_data.len() {
            let left_item_dist = (time - self.sorted_data[index - 1].time_stamp()).abs();
            if left_item_dist <= max_dist {
                return Some(IndexedItem {
                    index: index - 1,
                    item: &self.sorted_data[index - 1],
                });
            } else {
                return None;
            }
        }

        let right_item_dist = (self.sorted_data[index].time_stamp() - time).abs();

        if index == 0 {
            if right_item_dist <= max_dist {
                return Some(IndexedItem {
                    index,
                    item: &self.sorted_data[index],
                });
            } else {
                return None;
            }
        }

        let left_item_dist = (time - self.sorted_data[index - 1].time_stamp()).abs();

        if left_item_dist < right_item_dist {
            if left_item_dist <= max_dist {
                return Some(IndexedItem {
                    index: index - 1,
                    item: &self.sorted_data[index - 1],
                });
            } else {
                return None;
            }
        }

        if right_item_dist <= max_dist {
            Some(IndexedItem {
                index,
                item: &self.sorted_data[index],
            })
        } else {
            None
        }
    }

    /// Add an item to the time series.
    pub fn insert(&mut self, item: T) -> Option<()> {
        if !item.time_stamp().is_finite() {
            warn!(
                "TimeSeries::insert: skipping infinite time stamp {}",
                item.time_stamp()
            );
            return None;
        }
        // struct invariant: sorted_data is sorted by time_stamp

        if self.sorted_data.is_empty() || self.newest().unwrap().time_stamp() <= item.time_stamp() {
            // Common case for items inserted in order:
            //
            // TimeSeries is empty or item is newer than the last item. We can just add it
            // to the end, and the time series remains sorted.
            self.sorted_data.push_back(item);
            return Some(());
        }

        let index = self
            .sorted_data
            .partition_point(|probe| probe.time_stamp() < item.time_stamp());
        // insert item at index, so that time series remains sorted
        self.sorted_data.insert(index, item);

        Some(())
    }

    /// Number of items in the time series.
    pub fn len(&self) -> usize {
        self.sorted_data.len()
    }

    /// Check if the time series is empty.
    pub fn is_empty(&self) -> bool {
        self.sorted_data.is_empty()
    }

    /// Get the first in the time series.
    pub fn oldest(&self) -> Option<&T> {
        self.sorted_data.front()
    }

    /// Get the last in the time series.
    pub fn newest(&self) -> Option<&T> {
        self.sorted_data.back()
    }

    /// get time interval of time series
    pub fn time_interval(&self) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        Interval::from_bounds(
            self.oldest().unwrap().time_stamp(),
            self.newest().unwrap().time_stamp(),
        )
    }

    /// Prune older data by removing all elements older than newest minus `duration`.
    pub fn prune_older_than(&mut self, duration: Duration) {
        if self.is_empty() {
            return;
        }

        // We want to remove all elements older keep_time_point.
        let keep_time_point = self.newest().unwrap().time_stamp() - duration.as_secs_f64();
        // The n-th element is the first item >= keep_time_point.
        let n = self
            .sorted_data
            .partition_point(|x| x.time_stamp() < keep_time_point);

        self.prune_n_oldest(n);
    }

    /// removes specified range of elements
    pub fn drain<R: core::ops::RangeBounds<usize>>(&mut self, range: R) {
        self.sorted_data.drain(range);
    }

    /// removes N elements from the front
    pub fn prune_n_oldest(&mut self, n: usize) {
        self.sorted_data.drain(0..n);
    }

    /// clears the time series
    pub fn clear(&mut self) {
        self.sorted_data.clear();
    }
}

impl<T: HasInterpolate> TimeSeries<T> {
    /// Interpolate between two items in the time series.
    pub fn interpolate(&self, time: f64) -> Option<T> {
        let index = self
            .sorted_data
            .partition_point(|probe| probe.time_stamp() < time);

        if index == 0 {
            let oldest = self.oldest().unwrap();
            if oldest.time_stamp() == time {
                return Some(oldest.clone());
            }
            return None;
        }
        if index == self.sorted_data.len() {
            let newest = self.newest().unwrap();
            if newest.time_stamp() == time {
                return Some(newest.clone());
            }
            return None;
        }

        let left_item = &self.sorted_data[index - 1];
        let right_item = &self.sorted_data[index];

        let w =
            (time - left_item.time_stamp()) / (right_item.time_stamp() - left_item.time_stamp());

        Some(left_item.interpolate(right_item, w))
    }
}

impl<T: HasTimeStamp> IntoIterator for TimeSeries<T> {
    type Item = T;
    type IntoIter = alloc::collections::vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.sorted_data.into_iter()
    }
}

impl<'a, T: HasTimeStamp> IntoIterator for &'a TimeSeries<T> {
    type Item = &'a T;
    type IntoIter = alloc::collections::vec_deque::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.sorted_data.iter()
    }
}

impl<T: HasTimeStamp> Index<usize> for TimeSeries<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.sorted_data[index]
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use super::*;

    // Mock struct implementing HasTimeStamp
    #[derive(Debug, PartialEq, Clone)]
    struct TestItem {
        timestamp: f64,
        value: f64,
    }

    impl HasTimeStamp for TestItem {
        fn time_stamp(&self) -> f64 {
            self.timestamp
        }
    }

    impl HasInterpolate for TestItem {
        fn interpolate(&self, other: &Self, w: f64) -> Self {
            let value = self.value + (other.value - self.value) * w;
            TestItem {
                timestamp: self.timestamp + (other.timestamp - self.timestamp) * w,
                value,
            }
        }
    }

    #[test]
    fn test_insert_and_len() {
        let mut series = TimeSeries::new();
        assert_eq!(series.len(), 0);

        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        assert_eq!(series.len(), 1);

        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });
        assert_eq!(series.len(), 2);
    }

    #[test]
    fn test_insert_in_order() {
        let mut series = TimeSeries::<TestItem>::new();
        let items = [
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 3.0,
                value: 30.0,
            },
        ];

        for item in items.iter() {
            assert!(series.insert(item.clone()).is_some());
        }

        assert_eq!(series.len(), 3);
        for (i, item) in series.sorted_data.iter().enumerate() {
            assert_eq!(item, &items[i]);
        }
    }

    #[test]
    fn test_insert_out_of_order() {
        let mut series = TimeSeries::new();
        let items = [
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 3.0,
                value: 30.0,
            },
            TestItem {
                timestamp: 1.5,
                value: 15.0,
            },
        ];

        for item in &items {
            assert!(series.insert(item.clone()).is_some());
        }

        assert_eq!(series.len(), 4);
        let expected_order = [
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 1.5,
                value: 15.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 3.0,
                value: 30.0,
            },
        ];

        for (i, item) in series.sorted_data.iter().enumerate() {
            assert_eq!(*item, expected_order[i]);
        }
    }

    #[test]
    fn test_insert_duplicate_timestamps() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 1.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 30.0,
            },
        ];

        for item in &items {
            assert!(series.insert(item.clone()).is_some());
        }

        assert_eq!(series.len(), 3);
        let expected_order = [
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 1.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 30.0,
            },
        ];

        for (i, item) in series.sorted_data.iter().enumerate() {
            assert_eq!(*item, expected_order[i]);
        }
    }

    #[test]
    fn test_insert_infinite_timestamp() {
        let mut series = TimeSeries::new();
        let finite_item = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        let infinite_item = TestItem {
            timestamp: f64::INFINITY,
            value: 20.0,
        };

        assert!(series.insert(finite_item.clone()).is_some());
        assert!(series.insert(infinite_item.clone()).is_none()); // Should be skipped

        assert_eq!(series.len(), 1);
        assert_eq!(series.sorted_data[0], finite_item);
    }

    #[test]
    fn test_insert_with_non_finite_timestamp() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: f64::INFINITY,
            value: 10.0,
        });
        assert_eq!(series.len(), 0); // Shouldn't add item

        series.insert(TestItem {
            timestamp: f64::NAN,
            value: 10.0,
        });
        assert_eq!(series.len(), 0); // Shouldn't add item
    }

    #[test]
    fn test_prune_older_than() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 3.0,
            value: 10.0,
        });

        series.prune_older_than(Duration::from_secs_f64(1.1));
        assert_eq!(series.len(), 2); // Should remove the first item
        assert_eq!(series.oldest().unwrap().time_stamp(), 2.0);
    }

    #[test]
    fn test_time_interval() {
        let mut series = TimeSeries::new();
        assert!(series.time_interval().is_empty());

        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 3.0,
            value: 10.0,
        });

        let interval = series.time_interval();
        assert_eq!(interval.try_lower().unwrap(), 1.0);
        assert_eq!(interval.try_upper().unwrap(), 3.0);
    }
    #[test]
    fn test_find_nearest_empty_series() {
        let series: TimeSeries<TestItem> = TimeSeries::new();
        let result = series.find_nearest(1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_nearest_single_item() {
        let mut series = TimeSeries::new();
        let item = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        series.insert(item.clone());

        // Query for the exact timestamp
        let result = series.find_nearest(1.0).unwrap();
        assert_eq!(result.index, 0);
        assert_eq!(result.item, &item);

        // Query for a different timestamp (should still find the only item)
        let result = series.find_nearest(2.0).unwrap();
        assert_eq!(result.index, 0);
        assert_eq!(result.item, &item);
    }

    #[test]
    fn test_find_nearest_within_empty() {
        let series: TimeSeries<TestItem> = TimeSeries::new();
        let result = series.find_nearest_within(1.0, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_nearest_within_single_item_within() {
        let mut series = TimeSeries::new();
        let item = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        series.insert(item.clone());

        let result = series.find_nearest_within(1.0, 0.5);
        assert!(result.is_some());
        let indexed = result.unwrap();
        assert_eq!(indexed.index, 0);
        assert_eq!(indexed.item, &item);
    }

    #[test]
    fn test_find_nearest_within_single_item() {
        let mut series = TimeSeries::new();
        let item = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        series.insert(item.clone());

        let result = series.find_nearest_within(2.0, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_nearest_within_multiple_items_left_closer() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 4.0,
                value: 40.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        // Query at 3.0, max_dist 1.5
        // Closest is 2.0 (distance 1.0) vs 4.0 (distance 1.0)
        // Should return the left one (2.0)
        let result = series.find_nearest_within(2.9, 1.5);
        assert!(result.is_some());
        let indexed = result.unwrap();
        assert_eq!(indexed.index, 1);
        assert_eq!(indexed.item, &items[1]);
    }

    #[test]
    fn test_find_nearest_within_multiple_items_right_closer() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 4.0,
                value: 40.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        // Query at 3.5, max_dist 1.0
        // Closest is 4.0 (distance 0.5) vs 2.0 (distance 1.5)
        // Should return the right one (4.0)
        let result = series.find_nearest_within(3.5, 1.0);
        assert!(result.is_some());
        let indexed = result.unwrap();
        assert_eq!(indexed.index, 2);
        assert_eq!(indexed.item, &items[2]);
    }

    #[test]
    fn test_find_nearest_within_exact_match() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 3.0,
                value: 30.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        // Query exact timestamp
        let result = series.find_nearest_within(2.0, 0.0);
        assert!(result.is_some());
        let indexed = result.unwrap();
        assert_eq!(indexed.index, 1);
        assert_eq!(indexed.item, &items[1]);
    }

    #[test]
    fn test_find_nearest_within_ouserieside_max_dist() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            },
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            },
            TestItem {
                timestamp: 3.0,
                value: 30.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        // Query at 4.0 with max_dist 0.5 (closest is 3.0, distance 1.0)
        let result = series.find_nearest_within(4.0, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_nearest_within_infinite_time() {
        let mut series = TimeSeries::new();
        let item = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        series.insert(item.clone());

        let result = series.find_nearest_within(f64::INFINITY, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_prune_older_than_empty() {
        let mut series: TimeSeries<TestItem> = TimeSeries::new();
        series.prune_older_than(Duration::from_secs(10));
        assert!(series.is_empty());
    }

    #[test]
    fn test_prune_older_than_no_prune() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 10.0,
                value: 100.0,
            },
            TestItem {
                timestamp: 20.0,
                value: 200.0,
            },
            TestItem {
                timestamp: 30.0,
                value: 300.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        // Duration is 5 seconds, keep_time_point = 30 - 5 = 25
        // Should remove items with timestamp < 25, i.e., 10 and 20
        series.prune_older_than(Duration::from_secs(5));

        assert_eq!(series.len(), 1);
        assert_eq!(series.sorted_data[0], items[2]);
    }

    #[test]
    fn test_prune_older_than_some_prune() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 10.0,
                value: 100.0,
            },
            TestItem {
                timestamp: 20.0,
                value: 200.0,
            },
            TestItem {
                timestamp: 30.0,
                value: 300.0,
            },
            TestItem {
                timestamp: 40.0,
                value: 400.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        // Duration is 15 seconds, keep_time_point = 40 - 15 = 25
        // Should remove items with timestamp < 25, i.e., 10 and 20
        series.prune_older_than(Duration::from_secs(15));

        assert_eq!(series.len(), 2);
        assert_eq!(series.sorted_data[0], items[2]);
        assert_eq!(series.sorted_data[1], items[3]);
    }

    #[test]
    fn test_time_interval_empty() {
        let series: TimeSeries<TestItem> = TimeSeries::new();
        let interval = series.time_interval();
        assert_eq!(interval, Interval::empty());
    }

    #[test]
    fn test_time_interval_non_empty() {
        let mut series = TimeSeries::new();
        let items = alloc::vec![
            TestItem {
                timestamp: 5.0,
                value: 50.0,
            },
            TestItem {
                timestamp: 10.0,
                value: 100.0,
            },
            TestItem {
                timestamp: 15.0,
                value: 150.0,
            },
        ];

        for item in &items {
            series.insert(item.clone());
        }

        let interval = series.time_interval();
        assert_eq!(interval, Interval::from_bounds(5.0, 15.0));
    }

    #[test]
    fn test_oldest_and_newest() {
        let mut series = TimeSeries::new();
        assert!(series.oldest().is_none());
        assert!(series.newest().is_none());

        let item1 = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        series.insert(item1.clone());
        assert_eq!(series.oldest(), Some(&item1));
        assert_eq!(series.newest(), Some(&item1));

        let item2 = TestItem {
            timestamp: 2.0,
            value: 20.0,
        };
        series.insert(item2.clone());
        assert_eq!(series.oldest(), Some(&item1));
        assert_eq!(series.newest(), Some(&item2));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut series = TimeSeries::new();
        assert_eq!(series.len(), 0);
        assert!(series.is_empty());

        let item = TestItem {
            timestamp: 1.0,
            value: 10.0,
        };
        series.insert(item.clone());
        assert_eq!(series.len(), 1);
        assert!(!series.is_empty());

        series.insert(item.clone());
        assert_eq!(series.len(), 2);
    }

    #[test]
    fn test_into_iter_owned() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });
        series.insert(TestItem {
            timestamp: 3.0,
            value: 30.0,
        });

        // Test IntoIterator for owned TimeSeries
        let mut iter = series.into_iter();
        assert_eq!(
            iter.next(),
            Some(TestItem {
                timestamp: 1.0,
                value: 10.0,
            })
        );
        assert_eq!(
            iter.next(),
            Some(TestItem {
                timestamp: 2.0,
                value: 20.0,
            })
        );
        assert_eq!(
            iter.next(),
            Some(TestItem {
                timestamp: 3.0,
                value: 30.0,
            })
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_borrowed() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });
        series.insert(TestItem {
            timestamp: 3.0,
            value: 30.0,
        });

        // Test IntoIterator for borrowed TimeSeries
        let series_ref = &series;
        let mut iter = series_ref.into_iter();
        assert_eq!(
            iter.next(),
            Some(&TestItem {
                timestamp: 1.0,
                value: 10.0,
            })
        );
        assert_eq!(
            iter.next(),
            Some(&TestItem {
                timestamp: 2.0,
                value: 20.0,
            })
        );
        assert_eq!(
            iter.next(),
            Some(&TestItem {
                timestamp: 3.0,
                value: 30.0,
            })
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_index_access() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });
        series.insert(TestItem {
            timestamp: 3.0,
            value: 30.0,
        });

        // Test Index access
        assert_eq!(
            series[0],
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            }
        );
        assert_eq!(
            series[1],
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            }
        );
        assert_eq!(
            series[2],
            TestItem {
                timestamp: 3.0,
                value: 30.0,
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });

        // Accessing out-of-bounds index should panic
        let _ = series[2];
    }

    #[test]
    fn test_interpolate_simple_case() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });

        // Interpolating exactly between the two items (w = 0.5)
        let interpolated = series.interpolate(1.5).unwrap();
        assert_eq!(
            interpolated,
            TestItem {
                timestamp: 1.5,
                value: 15.0,
            }
        );
    }

    #[test]
    fn test_interpolate_exact_left_item() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });

        // Interpolating at the left boundary (w = 0.0)
        let interpolated = series.interpolate(1.0).unwrap();
        assert_eq!(
            interpolated,
            TestItem {
                timestamp: 1.0,
                value: 10.0,
            }
        );
    }

    #[test]
    fn test_interpolate_exact_right_item() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });

        // Interpolating at the right boundary (w = 1.0)
        let interpolated = series.interpolate(2.0).unwrap();
        assert_eq!(
            interpolated,
            TestItem {
                timestamp: 2.0,
                value: 20.0,
            }
        );
    }

    #[test]
    fn test_interpolate_outside_bounds() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });

        // Trying to interpolate before the first item (should return None)
        assert!(series.interpolate(0.5).is_none());

        // Trying to interpolate after the last item (should return None)
        assert!(series.interpolate(2.5).is_none());
    }

    #[test]
    fn test_interpolate_with_multiple_items() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 2.0,
            value: 20.0,
        });
        series.insert(TestItem {
            timestamp: 3.0,
            value: 30.0,
        });

        // Interpolating between the second and third item (w = 0.5 between 2.0 and 3.0)
        let interpolated = series.interpolate(2.5).unwrap();
        assert_eq!(
            interpolated,
            TestItem {
                timestamp: 2.5,
                value: 25.0,
            }
        );
    }

    #[test]
    fn test_interpolate_with_close_items() {
        let mut series = TimeSeries::new();
        series.insert(TestItem {
            timestamp: 1.0,
            value: 10.0,
        });
        series.insert(TestItem {
            timestamp: 1.1,
            value: 11.0,
        });

        // Interpolating very close to the first item
        let interpolated = series.interpolate(1.05).unwrap();
        assert_eq!(
            interpolated,
            TestItem {
                timestamp: 1.05,
                value: 10.5,
            }
        );
    }
}
