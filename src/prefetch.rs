//! One CPU-produced batch ahead of the consumer. No CUDA objects cross threads.
use anyhow::{Context, Result};
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
};

pub struct Prefetch<T> {
    requests: Option<SyncSender<()>>,
    ready: Option<Receiver<T>>,
    worker: Option<JoinHandle<usize>>,
}

impl<T: Send + 'static> Prefetch<T> {
    pub fn new(mut make: impl FnMut() -> T + Send + 'static) -> Result<Self> {
        let (requests, requested) = mpsc::sync_channel(1);
        // Rendezvous: the producer cannot queue more than one completed batch.
        let (output, ready) = mpsc::sync_channel(0);
        let worker = thread::Builder::new()
            .name("phrase-prefetch".into())
            .spawn(move || {
                while requested.recv().is_ok() {
                    let batch = make();
                    if output.send(batch).is_err() {
                        return 1;
                    }
                }
                0
            })
            .context("starting phrase producer")?;
        requests.send(()).context("starting first batch")?;
        Ok(Self {
            requests: Some(requests),
            ready: Some(ready),
            worker: Some(worker),
        })
    }

    pub fn take(&mut self) -> Result<T> {
        let batch = self
            .ready
            .as_ref()
            .unwrap()
            .recv()
            .context("phrase producer stopped unexpectedly")?;
        // Start exactly one successor while the caller processes this batch.
        self.requests
            .as_ref()
            .unwrap()
            .send(())
            .context("requesting next phrase batch")?;
        Ok(batch)
    }
}

impl<T> Prefetch<T> {
    pub fn finish(&mut self) -> Result<usize> {
        // Unblock either recv(request) or send(batch) before joining. Dropping a
        // pending PreparedBatch clears its recovery material through Zeroize.
        self.requests.take();
        self.ready.take();
        match self.worker.take() {
            Some(worker) => worker
                .join()
                .map_err(|_| anyhow::anyhow!("phrase producer panicked")),
            None => Ok(0),
        }
    }
}

impl<T> Drop for Prefetch<T> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Tracked(usize, Arc<AtomicUsize>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn preserves_order_and_drops_inflight_batch_on_early_stop() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let produced = Arc::new(AtomicUsize::new(0));
        let (started, observe) = mpsc::channel();
        let drops = dropped.clone();
        let count = produced.clone();
        let mut prefetch = Prefetch::new(move || {
            let index = count.fetch_add(1, Ordering::SeqCst);
            started.send(index).unwrap();
            Tracked(index, drops.clone())
        })
        .unwrap();
        for expected in 0..3 {
            let batch = prefetch.take().unwrap();
            assert_eq!(batch.0, expected);
            drop(batch);
        }
        // Wait for the one allowed successor; no sleeps or timing assumptions.
        while observe.recv().unwrap() != 3 {}
        assert_eq!(produced.load(Ordering::SeqCst), 4);
        assert_eq!(prefetch.finish().unwrap(), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn worker_failure_is_reported_and_joined() {
        let mut prefetch = Prefetch::<()>::new(|| panic!("test producer failure")).unwrap();
        assert!(prefetch.take().is_err());
        assert!(prefetch.finish().is_err());
    }
}
