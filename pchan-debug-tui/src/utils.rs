use std::iter::Peekable;

use pchan_utils::IgnorePoison;
use smol::lock::{RwLockReadGuard, RwLockWriteGuard};

pub struct InsertBetween<I, F, G, T>
where
    I: Iterator<Item = T>,
{
    iter: Peekable<I>,
    f: F,
    g: G,
    insert_next: Option<T>,
}

impl<I, F, G, T> Iterator for InsertBetween<I, F, G, T>
where
    I: Iterator<Item = T>,
    T: Clone,
    F: Fn(&T, &T) -> bool,
    G: Fn() -> T,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if let Some(x) = self.insert_next.take() {
            return Some(x);
        }
        let cur = self.iter.next()?;
        if let Some(next) = self.iter.peek() {
            if (self.f)(&cur, next) {
                self.insert_next = Some((self.g)());
            }
        }
        Some(cur)
    }
}

pub fn insert_between<I, F, G, T>(iter: I, f: F, g: G) -> InsertBetween<I::IntoIter, F, G, T>
where
    I: IntoIterator<Item = T>,
    T: Clone,
    F: Fn(&T, &T) -> bool,
    G: Fn() -> T,
{
    InsertBetween {
        iter: iter.into_iter().peekable(),
        f,
        g,
        insert_next: None,
    }
}

pub trait InsertBetweenExt<I, F, G, T>
where
    I: IntoIterator<Item = T>,
{
    fn insert_between(self, pred: F, value: G) -> InsertBetween<I::IntoIter, F, G, T>;
}

impl<I, F, G, T> InsertBetweenExt<I, F, G, T> for I
where
    I: IntoIterator<Item = T>,
    T: Clone,
    F: Fn(&T, &T) -> bool,
    G: Fn() -> T,
{
    fn insert_between(
        self,
        pred: F,
        value: G,
    ) -> InsertBetween<<I as IntoIterator>::IntoIter, F, G, T> {
        insert_between(self, pred, value)
    }
}
