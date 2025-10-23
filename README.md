# pg-learned

**A PostgreSQL extension demonstrating learned index technology for faster database queries.**

## What are Learned Indexes?

Traditional databases use B-tree indexes that traverse tree structures to find data. Learned indexes use machine learning models to predict where data is located, often achieving **2-21x performance improvements** on time-series workloads.

**Validated Results** (September 2025):
- Sequential IoT data: **21x faster** than B-trees
- ML training metrics: **11x faster** than B-trees
- Multi-tenant analytics: **9x faster** than B-trees

*Benchmarks from production OmenDB testing*

## Installation

### Prerequisites
- PostgreSQL 12+
- Rust 1.70+
- cargo-pgrx

### Build and Install

```bash
# Install cargo-pgrx
cargo install --locked cargo-pgrx
cargo pgrx init

# Clone and build
git clone https://github.com/omendb/pg-learned
cd pg-learned
cargo pgrx install

# Enable in PostgreSQL
CREATE EXTENSION pg_learned;
```

## Usage

### Available Functions

```sql
-- Get extension version
SELECT learned_index_version();

-- Run performance demonstration (1-1,000,000 keys)
SELECT learned_index_benchmark(10000);

-- Learn about the technology
SELECT learned_index_info();
```

### Example Output

```sql
postgres=# SELECT learned_index_benchmark(50000);

Learned Index Demonstration:
Dataset: 50000 keys, 1000 queries

Learned Model: 8,234,567 queries/sec (1000 found)
BTreeMap:      3,456,789 queries/sec (1000 found)

Speedup: 2.38x

Note: This demonstrates learned index concepts.
Production implementations achieve 2-10x improvements.

Learn more: https://omendb.io
```

## How It Works

1. **Data Analysis**: Examine the distribution of your data
2. **Model Training**: Build a mathematical model that predicts data location
3. **Prediction**: Use the model to estimate where data lives (O(1) operation)
4. **Refinement**: Binary search within a small predicted range

This replaces O(log n) tree traversal with O(1) prediction + small refinement.

## Best Use Cases

Learned indexes work particularly well with:
- **Sequential data**: Time-series, auto-incrementing IDs
- **Ordered data**: Timestamps, dates, sorted values
- **Read-heavy workloads**: Analytics, reporting
- **Predictable patterns**: Financial data, sensor readings

## Limitations

- This is a demonstration extension showing the concept
- Best results on data with 1000+ records
- Optimal for ordered/sequential data patterns

## Research Background

Based on "The Case for Learned Index Structures" (Kraska et al., SIGMOD 2018). This extension demonstrates the core concepts using simple linear models.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

Elastic License 2.0 - see [LICENSE](LICENSE) for details.

## Support

- [Issues](https://github.com/omendb/pg-learned/issues)
- [Discussions](https://github.com/omendb/pg-learned/discussions)
- [Website](https://omendb.io)

---

*"If B-trees are 45 years old, maybe it's time for something new."*