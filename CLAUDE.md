# pg-learned Developer Guide

## Overview
PostgreSQL extension demonstrating learned index technology.
Open source educational tool showing how learned indexes can improve lookup performance.

## Repository
- **Name**: pg-learned
- **License**: MIT
- **Location**: github.com/omendb/pg-learned

## Current Implementation
```rust
// Demonstration functions
learned_index_benchmark(num_keys) -> Performance comparison
learned_index_version() -> Version info
learned_index_info() -> Educational content
```

## Build & Test
```bash
# Install pgrx (first time only)
cargo install cargo-pgrx
cargo pgrx init

# Build and run extension
cargo pgrx run  # Starts PostgreSQL with extension loaded

# Test in PostgreSQL
CREATE EXTENSION pg_learned;
SELECT learned_index_version();
SELECT learned_index_benchmark(10000);
SELECT learned_index_info();
```

## Code Quality Standards
- **Input validation**: Proper bounds checking and error handling
- **Clear output**: Professional, informative benchmark results
- **Performance**: Real demonstrations of learned index benefits
- **Educational**: Help users understand the technology

## Contributing
Contributions welcome! Areas for improvement:
- Enhanced learned index algorithms (LinearIndex, RMI variants)
- Additional benchmark scenarios
- Educational documentation and examples
- Performance optimizations

## Development Notes
- Maintain high code quality standards
- Focus on educational value and clarity
- Ensure benchmarks demonstrate real performance characteristics
- Keep documentation up-to-date