use puffin::GlobalFrameView;

/// Printer for puffin benchmark.
pub struct PuffinPrinter {
    frame_view: GlobalFrameView,
}

impl Default for PuffinPrinter {
    fn default() -> Self {
        Self::new()
    }
}

impl PuffinPrinter {
    /// Create new PuffinPrinter.
    ///
    /// Note: Needs to be created before any benchmarks are run, and `puffin::set_scopes_on(true)`
    /// needs to be called beforehand.
    pub fn new() -> Self {
        PuffinPrinter {
            frame_view: puffin::GlobalFrameView::default(),
        }
    }

    fn ms_from_ns(time_ns: puffin::NanoSecond) -> f64 {
        time_ns as f64 / 1.0e6
    }

    /// Prints latest frame.
    pub fn print_latest(&self, name: &str) -> Result<(), &str> {
        let view = self.frame_view.lock();
        let frame = view.latest_frame().ok_or("No frame available")?;
        println!(
            "\n=== {name} (frame #{}) — total {:.3} ms ===",
            frame.frame_index(),
            Self::ms_from_ns(frame.duration_ns())
        );

        let scopes = view.scope_collection();
        let unpacked = frame.unpacked().unwrap_or_else(|never| match never {});

        for (thread_info, stream_info) in &unpacked.thread_streams {
            println!(
                "\n[{} thread]",
                if thread_info.name.is_empty() {
                    "<unnamed>"
                } else {
                    &thread_info.name
                }
            );

            // Walk only the top-level scopes; the helper recurses into children.
            let r = puffin::Reader::from_start(&stream_info.stream);
            for item in r {
                Self::print_scope_tree(
                    &stream_info.stream,
                    scopes,
                    item.unwrap(),
                    0,
                    Self::ms_from_ns(frame.meta().range_ns.0),
                );
            }
        }
        Ok(())
    }

    // Recursively print one scope and its children.
    fn print_scope_tree(
        stream: &puffin::Stream,
        scopes: &puffin::ScopeCollection,
        scope: puffin::Scope<'_>,
        depth: usize,
        frame_start_time_ms: f64,
    ) {
        let name = scopes
            .fetch_by_id(&scope.id)
            .map(|d| d.name().as_ref().to_string())
            .unwrap_or_else(|| "<unknown>".to_string());

        let scope_start_time_ms = Self::ms_from_ns(scope.record.start_ns) - frame_start_time_ms;

        let duration_ms = Self::ms_from_ns(scope.record.duration_ns);

        println!(
            "{:indent$}{:25}: {:8.3} ms, [{:8.3} ms, {:8.3} ms] ",
            "",
            name,
            duration_ms,
            scope_start_time_ms,
            scope_start_time_ms + duration_ms,
            indent = 2 * depth,
        );

        // Sum children:
        let mut off = scope.child_begin_position;
        while off < scope.child_end_position {
            let mut r = puffin::Reader::with_offset(stream, off).unwrap();
            if let Some(Ok(child)) = r.next() {
                Self::print_scope_tree(stream, scopes, child, depth + 1, frame_start_time_ms);
                off = child.next_sibling_position;
            } else {
                break;
            }
        }
    }
}
