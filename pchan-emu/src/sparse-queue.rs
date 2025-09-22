#[derive(Debug, Default, Clone)]
pub struct SparseQueue<T> {
    inner: Vec<Option<T>>,
    empty: bool,
}

impl<T> SparseQueue<T> {
    pub fn new() -> Self {
        SparseQueue {
            inner: Vec::new(),
            empty: true,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        SparseQueue {
            inner: Vec::with_capacity(cap),
            empty: true,
        }
    }

    pub fn as_vec(&self) -> &Vec<Option<T>> {
        &self.inner
    }

    pub fn as_vec_mut(&mut self) -> &mut Vec<Option<T>> {
        &mut self.inner
    }

    pub fn is_empty(&self) -> bool {
        self.empty
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter().flatten()
    }

    pub fn retain_mut(&mut self, mut pred: impl FnMut(&mut T) -> bool) {
        let mut empty = true;
        for el in self.inner.iter_mut() {
            if let Some(value) = el {
                let keep = pred(value);
                if !keep {
                    *el = None;
                } else {
                    // Some(T) found and not removed by predicate
                    empty = false;
                }
            }
        }
        self.empty = empty;
    }

    pub fn decay(&mut self, mut pred: impl FnMut(&mut T) -> bool) {
        while !self.empty {
            self.retain_mut(&mut pred);
        }
    }

    pub fn push(&mut self, value: T) {
        self.inner.push(Some(value));
        self.empty = false;
    }
}
