#![feature(negative_impls)]

use std::cell::OnceCell;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread::{JoinHandle, Thread};

pub struct Executor<'a> {
    _life:  PhantomData<&'a ()>,
    thread: Thread,
}

impl<'a> !Send for Executor<'a> {}
impl<'a> !Sync for Executor<'a> {}

impl<'a> Executor<'a> {
    pub fn new() -> Self {
        Self {
            _life:  PhantomData,
            thread: std::thread::current(),
        }
    }
    fn create_waker(&self) -> Waker {
        unsafe {
            Waker::from_raw(RawWaker::new(
                std::ptr::from_ref(&self.thread) as *const (),
                &VTABLE,
            ))
        }
    }
    pub fn block_on<O>(&self, task: impl Future<Output = O> + 'a) -> O {
        let waker = self.create_waker();
        let mut ctx = Context::from_waker(&waker);
        let mut pinned = std::pin::pin!(task);
        loop {
            match pinned.as_mut().poll(&mut ctx) {
                Poll::Ready(res) => return res,
                Poll::Pending => {
                    std::thread::park();
                }
            }
        }
    }
}

thread_local! {
    static THREAD_EXEC: OnceCell<Executor<'static>> = const { OnceCell::new() };
}

pub fn block_on<O>(task: impl Future<Output = O>) -> O {
    THREAD_EXEC.with(|exec| exec.get_or_init(Executor::new).block_on(task))
}

pub async fn unblock<O: Send + 'static>(func: impl Fn() -> O + Send + 'static) -> O {
    Unblock::new(func).await
}

static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop_waker);

unsafe fn clone(data: *const ()) -> RawWaker {
    RawWaker::new(data, &VTABLE)
}

unsafe fn wake(data: *const ()) {
    let thread = unsafe { data.cast::<Thread>().as_ref() }.expect("invalid thread pointer");
    thread.unpark();
}

unsafe fn wake_by_ref(data: *const ()) {
    unsafe { wake(data) }
}

unsafe fn drop_waker(_: *const ()) {
    // has no owned data
}

impl<'a> Default for Executor<'a> {
    fn default() -> Self {
        Self::new()
    }
}

struct Unblock<O> {
    handle: Option<JoinHandle<O>>,
    waker:  Arc<Mutex<Option<Waker>>>,
}

impl<O> Unblock<O>
where
    O: Send + 'static,
{
    fn new(func: impl Fn() -> O + Send + 'static) -> Self {
        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let w = waker.clone();
        let handle = std::thread::spawn(move || {
            let waker = w;
            let res = func();
            if let Ok(waker) = waker.lock()
                && let Some(waker) = &*waker
            {
                waker.wake_by_ref();
            }
            res
        });
        Self {
            handle: Some(handle),
            waker,
        }
    }
}

impl<O> Future for Unblock<O> {
    type Output = O;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut waker = self.waker.lock().unwrap();
            *waker = Some(cx.waker().clone());
        }
        let Some(handle) = self.handle.as_mut() else {
            return Poll::Pending;
        };

        match handle.is_finished() {
            true => {
                let handle = std::mem::take(&mut self.handle);
                let Some(handle) = handle else {
                    return Poll::Pending;
                };
                Poll::Ready(handle.join().expect("failed to join thread"))
            }
            false => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use smol::Timer;

    use super::*;

    #[test]
    fn it_works() {
        let exec = Executor::new();
        exec.block_on(async {
            println!("hi");
            Timer::after(Duration::from_secs(1)).await;
            println!("hi again");
        });
    }
}
