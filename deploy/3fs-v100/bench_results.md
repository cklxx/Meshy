# usrbio put/get 带宽定位（T5b）

环境：V100 节点，rxe soft-RoCE 单机回环，RF=2 双副本，64KiB KV page，
单 client 串行 submit->wait，128 MiB（micro_op.py，数据预生成、计时只含 io）。

## 1. 单 op 大小扫描（单 ioring，每 op 一次提交并等完成）

| op 大小 | op 数 | 写 MiB/s | 读 MiB/s |
|---:|---:|---:|---:|
| 64 KiB | 2048 | **10.3** | 55 |
| 128 KiB | 1024 | 5.8 | 95 |
| 256 KiB | 512 | 11.7 | 161 |
| 512 KiB | 256 | 22.4 | 181 |
| 1 MiB | 128 | **86.6** | 181 |
| 2 MiB | 64 | 86.5 | 182 |
| 4 MiB | 32 | 87.0 | 180 |
| 8 MiB（64MiB 样本） | 8 | 130 | 178 |

生产 page_size=64（page_first_direct，3.5 MiB/op）实测写 83、读 456；
page_size=128（7 MiB/op）写 154、读 476。

## 2. 并发扫描（bench_hicache.py：numjobs 个独立 client/ioring + 线程池）

| pageKiB | entries | numjobs | 写 | 读 |
|---:|---:|---:|---|---|
| 64 | 8/16/32 | 1 | 10.5 / 2.8 / 2.8 | 178–205 |
| 64 | 8 | ≥2 | **失败** EAGAIN (errno 11) / OSError 5 | — |
| 64 | 任意 | ≥2 | "no more sqes" / "same cqe fetched twice" | — |

多 ioring 在单个 rxe 设备上并发直接打满 SQ/CQE（soft-RoCE 限制），
不是 SGLang 的问题，真实网卡有独立硬件队列不会这样。
