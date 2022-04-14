use crate::trace::Trace;
use alloc::{boxed::Box, vec::Vec};
use core::cell::Cell;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

struct GcState {
    stats: GcStats,
    config: GcConfig,
    boxes_start: Option<*mut GcBox<dyn Trace>>,
}

unsafe impl Send for GcState {}

impl Drop for GcState {
    fn drop(&mut self) {
        if !self.config.leak_on_drop {
            collect_garbage(self);
        }
        // We have no choice but to leak any remaining nodes that
        // might be referenced from other thread-local variables.
    }
}

// Whether or not the thread is currently in the sweep phase of garbage collection.
// During this phase, attempts to dereference a `Gc<T>` pointer will trigger a panic.
pub static GC_DROPPING: AtomicBool = AtomicBool::new(false);
struct DropGuard;
impl DropGuard {
    #[must_use]
    fn new() -> DropGuard {
        GC_DROPPING.store(true, Ordering::Relaxed);
        DropGuard
    }
}
impl Drop for DropGuard {
    fn drop(&mut self) {
        GC_DROPPING.store(false, Ordering::Relaxed);
    }
}
pub fn finalizer_safe() -> bool {
    !GC_DROPPING.load(Ordering::Relaxed)
}

// The garbage collector's internal state.
static GC_STATE: Mutex<GcState> = Mutex::new(GcState {
    stats: GcStats {
        bytes_allocated: 0,
        collections_performed: 0,
    },
    config: GcConfig {
        used_space_ratio: 0.7,
        threshold: 100,
        leak_on_drop: false,
    },
    boxes_start: None,
});

const MARK_MASK: usize = 1 << (usize::BITS - 1);
const ROOTS_MASK: usize = !MARK_MASK;
const ROOTS_MAX: usize = ROOTS_MASK; // max allowed value of roots

pub(crate) struct GcBoxHeader {
    roots: Cell<usize>, // high bit is used as mark flag
    next: Option<*mut GcBox<dyn Trace>>,
}

impl GcBoxHeader {
    #[inline]
    pub const fn new(next: Option<*mut GcBox<dyn Trace>>) -> Self {
        GcBoxHeader {
            roots: Cell::new(1), // unmarked and roots count = 1
            next,
        }
    }

    #[inline]
    pub fn roots(&self) -> usize {
        self.roots.get() & ROOTS_MASK
    }

    #[inline]
    pub fn inc_roots(&self) {
        let roots = self.roots.get();

        // abort if the count overflows to prevent `mem::forget` loops
        // that could otherwise lead to erroneous drops
        if (roots & ROOTS_MASK) < ROOTS_MAX {
            self.roots.set(roots + 1); // we checked that this wont affect the high bit
        } else {
            panic!("roots counter overflow");
        }
    }

    #[inline]
    pub fn dec_roots(&self) {
        self.roots.set(self.roots.get() - 1) // no underflow check
    }

    #[inline]
    pub fn is_marked(&self) -> bool {
        self.roots.get() & MARK_MASK != 0
    }

    #[inline]
    pub fn mark(&self) {
        self.roots.set(self.roots.get() | MARK_MASK)
    }

    #[inline]
    pub fn unmark(&self) {
        self.roots.set(self.roots.get() & !MARK_MASK)
    }
}

#[repr(C)] // to justify the layout computation in Gc::from_raw
pub(crate) struct GcBox<T: Trace + ?Sized + 'static> {
    header: GcBoxHeader,
    data: T,
}

impl<T: Trace> GcBox<T> {
    /// Allocates a garbage collected `GcBox` on the heap,
    /// and appends it to the thread-local `GcBox` chain.
    ///
    /// A `GcBox` allocated this way starts its life rooted.
    pub(crate) fn new(value: T) -> *mut Self {
        let mut st = GC_STATE.lock();

        // XXX We should probably be more clever about collecting
        if st.stats.bytes_allocated > st.config.threshold {
            collect_garbage(&mut st);

            if st.stats.bytes_allocated as f64
                > st.config.threshold as f64 * st.config.used_space_ratio
            {
                // we didn't collect enough, so increase the
                // threshold for next time, to avoid thrashing the
                // collector too much/behaving quadratically.
                st.config.threshold =
                    (st.stats.bytes_allocated as f64 / st.config.used_space_ratio) as usize
            }
        }

        let gcbox = Box::into_raw(Box::new(GcBox {
            header: GcBoxHeader::new(st.boxes_start.take()),
            data: value,
        }));

        st.boxes_start = Some(gcbox);

        // We allocated some bytes! Let's record it
        st.stats.bytes_allocated += mem::size_of::<GcBox<T>>();

        // Return the pointer to the newly allocated data
        gcbox
    }
}

impl<T: Trace + ?Sized> GcBox<T> {
    /// Returns `true` if the two references refer to the same `GcBox`.
    pub(crate) fn ptr_eq(this: &GcBox<T>, other: &GcBox<T>) -> bool {
        // Use .header to ignore fat pointer vtables, to work around
        // https://github.com/rust-lang/rust/issues/46139
        ptr::eq(&this.header, &other.header)
    }

    /// Marks this `GcBox` and marks through its data.
    pub(crate) unsafe fn trace_inner(&self) {
        if !self.header.is_marked() {
            self.header.mark();
            self.data.trace();
        }
    }

    /// Increases the root count on this `GcBox`.
    /// Roots prevent the `GcBox` from being destroyed by the garbage collector.
    pub(crate) unsafe fn root_inner(&self) {
        self.header.inc_roots();
    }

    /// Decreases the root count on this `GcBox`.
    /// Roots prevent the `GcBox` from being destroyed by the garbage collector.
    pub(crate) unsafe fn unroot_inner(&self) {
        self.header.dec_roots();
    }

    /// Returns a reference to the `GcBox`'s value.
    pub(crate) fn value(&self) -> &T {
        &self.data
    }
}

/// Collects garbage.
fn collect_garbage(st: &mut GcState) {
    st.stats.collections_performed += 1;

    struct Unmarked {
        incoming: *mut Option<*mut GcBox<dyn Trace>>,
        this: *mut GcBox<dyn Trace>,
    }
    unsafe fn mark(head: &mut Option<*mut GcBox<dyn Trace>>) -> Vec<Unmarked> {
        // Walk the tree, tracing and marking the nodes
        let mut mark_head = *head;
        while let Some(node) = mark_head {
            if (*node).header.roots() > 0 {
                (*node).trace_inner();
            }

            mark_head = (*node).header.next;
        }

        // Collect a vector of all of the nodes which were not marked,
        // and unmark the ones which were.
        let mut unmarked = Vec::new();
        let mut unmark_head = head;
        while let Some(node) = *unmark_head {
            if (*node).header.is_marked() {
                (*node).header.unmark();
            } else {
                unmarked.push(Unmarked {
                    incoming: unmark_head,
                    this: node,
                });
            }
            unmark_head = &mut (*node).header.next;
        }
        unmarked
    }

    unsafe fn sweep(finalized: Vec<Unmarked>, bytes_allocated: &mut usize) {
        let _guard = DropGuard::new();
        for node in finalized.into_iter().rev() {
            if (*node.this).header.is_marked() {
                continue;
            }
            let incoming = node.incoming;
            let mut node = Box::from_raw(node.this);
            *bytes_allocated -= mem::size_of_val::<GcBox<_>>(&node);
            *incoming = node.header.next.take();
        }
    }

    unsafe {
        let unmarked = mark(&mut st.boxes_start);
        if unmarked.is_empty() {
            return;
        }
        for node in &unmarked {
            Trace::finalize_glue(&(*node.this).data);
        }
        mark(&mut st.boxes_start);
        sweep(unmarked, &mut st.stats.bytes_allocated);
    }
}

/// Immediately triggers a garbage collection on the current thread.
///
/// This will panic if executed while a collection is currently in progress
pub fn force_collect() {
    let mut st = GC_STATE.lock();
    collect_garbage(&mut st);
}

pub struct GcConfig {
    pub threshold: usize,
    /// after collection we want the the ratio of used/total to be no
    /// greater than this (the threshold grows exponentially, to avoid
    /// quadratic behavior when the heap is growing linearly with the
    /// number of `new` calls):
    pub used_space_ratio: f64,
    /// For short-running processes it is not always appropriate to run
    /// GC, sometimes it is better to let system free the resources
    pub leak_on_drop: bool,
}

/* #[allow(dead_code)]
pub fn configure(configurer: impl FnOnce(&mut GcConfig)) {
    let mut st = GC_STATE.lock();
    configurer(&mut st.config);
} */

#[derive(Clone)]
pub struct GcStats {
    pub bytes_allocated: usize,
    pub collections_performed: usize,
}

/* #[allow(dead_code)]
pub fn stats() -> GcStats {
    GC_STATE.lock().stats.clone()
} */
