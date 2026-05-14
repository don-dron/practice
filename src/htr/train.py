from __future__ import annotations

from pathlib import Path
import functools
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import bisect

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from htr.charset import Charset, charset_from_strings
from htr.cuda_line_batch import coco_collate_mixed_lines, collate_gpu_lines_jpeg_cuda_batch
from htr.data.coco_lines import COCOLinesDataset, coco_collate_fn
from htr.data.page_txt_pairs import PageTxtPairsDataset, _normalize_document_text
from htr.data.split import Subset, random_split_indices
from htr.device import move_training_image_batch, pick_device
from htr.hardware_parallel import (
    AsyncCpuBatchPrefetcher,
    apply_main_thread_env_and_torch,
    effective_dataloader_worker_torch_threads,
    effective_prefetch_factor,
    hardware_profile,
)
from htr.eval.metrics import lev_ratio
from htr.io.checkpoint import save_checkpoint
from htr.models import resolve_model
from htr.models.attention_line import AttentionLineSeq2Seq
from htr.models.resnet_pretrained_line_ctc import PretrainedResnetLineCTC
from htr.transforms import TrainAugmentation


def _text_at_index_for_charset(ds: Dataset, idx: int) -> str:
    """Текст по индексу без декодирования изображений (иначе COCO на сотнях тысяч строк «висит» на старте)."""
    if isinstance(ds, Subset):
        return _text_at_index_for_charset(ds.ds, int(ds.indices[idx]))
    if isinstance(ds, ConcatDataset):
        i = int(idx)
        cums = ds.cumulative_sizes
        di = bisect.bisect_right(cums, i)
        off = i if di == 0 else i - int(cums[di - 1])
        return _text_at_index_for_charset(ds.datasets[di], off)
    if isinstance(ds, COCOLinesDataset):
        return str(ds.samples[int(idx)][1])
    if isinstance(ds, PageTxtPairsDataset):
        _, txt_path = ds.samples[int(idx)]
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        return _normalize_document_text(raw, ds.max_text_chars)
    item = ds[int(idx)]
    return str(item["text"])


def _collect_charset_texts(full_ds: Dataset, train_ix: list[int], tc: dict) -> List[str]:
    """Чтение подписей для алфавита; при parallel_charset_text_reads или hardware_utilization=max — пул потоков."""
    n = len(train_ix)
    prof = hardware_profile(tc)
    use_parallel = bool(tc.get("parallel_charset_text_reads", False)) or prof == "max"
    if not use_parallel or n < 4096:
        if n > 80_000:
            _it = tqdm(train_ix, desc="[htr-train] тексты для Charset", mininterval=2.0, unit="стр")
        else:
            _it = train_ix
        return [_text_at_index_for_charset(full_ds, int(i)) for i in _it]

    cpu = os.cpu_count() or 8
    mw = min(32, max(4, cpu // 2))
    chunk = max(128, n // max(mw * 32, 1))

    def one(ii: int) -> str:
        return _text_at_index_for_charset(full_ds, int(ii))

    with ThreadPoolExecutor(max_workers=mw) as ex:
        if n > 80_000:
            return list(
                tqdm(
                    ex.map(one, train_ix, chunksize=chunk),
                    total=n,
                    desc="[htr-train] тексты для Charset",
                    mininterval=2.0,
                    unit="стр",
                )
            )
        return list(ex.map(one, train_ix, chunksize=chunk))


def _make_grad_scaler(enabled: bool):
    """GradScaler: torch.amp API on newer PyTorch, legacy import on older builds."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    from torch.cuda.amp import GradScaler as _LegacyGradScaler

    return _LegacyGradScaler(enabled=enabled)


def _autocast_cuda():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    import torch.cuda.amp as cuda_amp

    return cuda_amp.autocast(enabled=True)


def _pack_targets(texts_batch: list[str], charset: Charset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [charset.encode(t) for t in texts_batch]
    tlens = torch.tensor([len(x) for x in tensors], dtype=torch.long, device=device)
    if tlens.sum().item() == 0:
        return torch.zeros(0, dtype=torch.long, device=device), tlens
    concat = torch.tensor([idx for seq in tensors for idx in seq], dtype=torch.long, device=device)
    return concat, tlens


def _optional_positive_int(raw: object) -> Optional[int]:
    if raw is None:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _maybe_torch_compile(model: nn.Module, tc: dict, device: torch.device) -> nn.Module:
    if device.type != "cuda":
        return model
    if not bool(tc.get("torch_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        print("[htr-train] training.torch_compile=true, но torch.compile недоступен — пропуск")
        return model
    mode = tc.get("torch_compile_mode", "default")
    if not isinstance(mode, str):
        mode = "default"
    print(f"[htr-train] torch.compile(mode={mode!r}, dynamic=True if supported); первый запуск может быть дольше")
    compile_fn = torch.compile  # type: ignore[attr-defined]
    try:
        return compile_fn(model, mode=mode, dynamic=True)  # type: ignore[misc]
    except TypeError:
        try:
            return compile_fn(model, mode=mode)
        except Exception as ex:
            print(f"[htr-train] torch.compile не удался ({ex!r}) — обучение без компиляции")
            return model


def _du_sh_quick(cache_root: Path) -> Optional[str]:
    """Размер каталога через du -sh без обхода каждого файла в Python."""
    import shutil
    import subprocess

    du_bin = shutil.which("du")
    if not du_bin:
        return None
    try:
        proc = subprocess.run(
            [du_bin, "-sh", str(cache_root)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.split()[0].strip()
    except Exception:
        pass
    return None


def _log_disk_cache_readiness(parts: list, kinds: list[str], counts: list[int]) -> None:
    """Подсказки по заполнению preprocessed_cache: быстрый du + опционально полный счётчик .pt."""
    idxs = [i for i, p in enumerate(parts) if getattr(p, "preprocessed_cache_root", None) is not None]
    if not idxs:
        return

    for i in idxs:
        p = parts[i]
        root = Path(p.preprocessed_cache_root)  # type: ignore[arg-type,misc]
        if not root.is_dir():
            print(f"[htr-train] кэш: каталог ещё не создан или пуст — {root}")
            continue
        human = _du_sh_quick(root)
        h = human if human is not None else "(нет команды du — смотрите размер папки вручную)"
        print(f"[htr-train] кэш на диске ({kinds[i]}): ~{h}\t{root}")
        print(f"[htr-train]   цель: ~{counts[i]} файлов .pt (один на пример); пока кэш не полный, CPU будет высоким при декоде.")
    print(
        "[htr-train] «Прогрев» диска: после полного прохода по train размер подкаталога почти не растёт, число .pt → ожидаемому. "
        "Точный подсчёт (долго): HTR_DISK_CACHE_STATS=1 … или ./scripts/disk_cache_stats.sh"
    )

    ev = os.environ.get("HTR_DISK_CACHE_STATS", "").strip().lower()
    if ev not in ("1", "true", "yes"):
        return

    import time

    for i in idxs:
        p = parts[i]
        root = Path(p.preprocessed_cache_root)  # type: ignore[arg-type,misc]
        if not root.is_dir():
            continue
        exp = counts[i]
        t0 = time.perf_counter()
        n_pt = sum(1 for _ in root.rglob("*.pt"))
        dt = time.perf_counter() - t0
        pct = 100.0 * n_pt / max(1, exp)
        print(
            f"[htr-train] HTR_DISK_CACHE_STATS: {kinds[i]} — {n_pt}/{exp} .pt (~{pct:.1f}%) за {dt:.1f}s"
        )


def _training_objective(cfg: dict) -> str:
    return str(cfg.get("training", {}).get("objective", "ctc")).strip().lower()


def _decoder_max_steps(cfg: dict) -> int:
    return max(16, int(cfg.get("training", {}).get("decoder_max_steps", 512)))


def _freeze_from_plan(cfg: dict) -> int:
    ppt = cfg.get("planned_transfer_policy")
    if not isinstance(ppt, dict):
        return 0
    fx = ppt.get("freeze_encoder_until_epoch")
    if fx is None:
        return 0
    try:
        return max(0, int(fx))
    except (TypeError, ValueError):
        return 0


def _freeze_backbone_epochs(cfg: dict) -> int:
    mh = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    fv = mh.get("freeze_backbone_epochs")
    if fv is not None:
        try:
            return max(0, int(fv))
        except (TypeError, ValueError):
            return _freeze_from_plan(cfg)
    return _freeze_from_plan(cfg)


def _set_pretrained_backbone_frozen(model: nn.Module, backbone_frozen: bool) -> None:
    if not isinstance(model, PretrainedResnetLineCTC):
        return
    for p in model.backbone_parameters():
        p.requires_grad = not backbone_frozen


def _truncate_enc(enc: list[int], cap_steps: int) -> list[int]:
    steps = len(enc) + 1 if enc else 1
    if steps <= cap_steps:
        return enc
    return enc[: max(0, cap_steps - 1)]


def _prepare_attention_batches(
    model: AttentionLineSeq2Seq,
    charset: Charset,
    texts_batch: list[str],
    device: torch.device,
    decoder_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sos = model.sos_idx_emb
    pad_emb = model.pad_idx_emb
    eos_c = model.eos_logits

    batch_enc: list[list[int]] = []
    lengths = []
    for tex in texts_batch:
        ee = charset.encode(tex)
        ee_t = _truncate_enc(ee, decoder_cap)
        batch_enc.append(ee_t)
        ln = len(ee_t)
        lengths.append(1 if ln == 0 else ln + 1)

    lm = max(lengths)
    b_sz = len(texts_batch)

    tin = torch.full((b_sz, lm), pad_emb, dtype=torch.long, device=device)
    targ = torch.full((b_sz, lm), -100, dtype=torch.long, device=device)
    msk = torch.zeros(b_sz, lm, dtype=torch.bool, device=device)

    for bi, enc_ids in enumerate(batch_enc):
        if not enc_ids:
            msk[bi, 0] = True
            tin[bi, 0] = sos
            targ[bi, 0] = eos_c
            continue

        steps = len(enc_ids) + 1
        for s in range(steps):
            msk[bi, s] = True
            tin[bi, s] = sos if s == 0 else enc_ids[s - 1] - 1
            if s < len(enc_ids):
                targ[bi, s] = enc_ids[s] - 1
            else:
                targ[bi, s] = eos_c

    return tin, targ, msk


def _attention_hypothesis(ce_tokens: list[int], charset: Charset) -> str:
    out: list[str] = []
    for k in ce_tokens:
        cid = k + 1
        if 1 <= cid < len(charset.itos):
            out.append(charset.itos[cid])
    return "".join(out)


class _DataloaderWorkerTorchThreadsInit:
    """Ограничивает BLAS/OpenMP/torch внутри воркера DataLoader.

    Без этого num_workers процессов × десятки потоков OpenMP дают «шумный» CPU и пустую GPU.
    """

    __slots__ = ("t",)

    def __init__(self, torch_threads: int) -> None:
        self.t = max(1, int(torch_threads))

    def __call__(self, worker_id: int) -> None:
        del worker_id
        ts = str(self.t)
        os.environ["OMP_NUM_THREADS"] = ts
        os.environ["MKL_NUM_THREADS"] = ts
        os.environ["OPENBLAS_NUM_THREADS"] = ts
        os.environ["NUMEXPR_NUM_THREADS"] = ts
        os.environ["VECLIB_MAXIMUM_THREADS"] = ts
        try:
            torch.set_num_threads(int(self.t))
        except Exception:
            pass


def _google_colab_runtime() -> bool:
    """Среда Google Colab (мало CPU, Jupyter + multiprocessing без persistent_workers надёжнее)."""
    return bool(os.environ.get("COLAB_RELEASE_TAG"))


def _colab_default_dataloader_workers_cap() -> int:
    """Верхний предел workers на Colab без HTR_COLAB_NUM_WORKERS: не «глушить» в 2 на машинах с 4–12 ядрами."""
    cpu = os.cpu_count() or 4
    # Одно ядро оставить под главный процесс; не больше 8 — ограничение RAM/процессов типичного сеанса.
    return max(2, min(8, max(1, cpu - 1)))


def _windows_dataloader_fast(dc: dict) -> bool:
    """Временно агрессивнее workers/prefetch на Windows (риск ошибки 1455). YAML или env."""
    ev = os.environ.get("HTR_WIN_DATALOADER_FAST", "").strip().lower()
    if ev in ("1", "true", "yes"):
        return True
    return bool(dc.get("windows_dataloader_fast", False))


def _dataload_worker_count(dc: dict) -> int:
    nw_requested = int(dc.get("num_workers", 0))
    nw = max(0, nw_requested)
    if sys.platform == "win32":
        _cap_raw = os.environ.get("HTR_WIN_MAX_NUM_WORKERS", "").strip()
        if _cap_raw != "0":
            # Много воркеров + spawn + крупные батчи часто даёт RuntimeError 1455 (shared file mapping).
            fast = _windows_dataloader_fast(dc)
            _cap = 8 if fast else 4
            if _cap_raw != "":
                try:
                    _cap = max(1, min(32, int(_cap_raw)))
                except ValueError:
                    pass
            if nw > _cap:
                print(
                    f"[htr-train] num_workers: YAML asked {nw}, capping at {_cap} on Windows "
                    f"(избегает 1455; быстрый режим: windows_dataloader_fast / HTR_WIN_DATALOADER_FAST=1 кап↑8; "
                    f"HTR_WIN_MAX_NUM_WORKERS=N; без капа: =0; стабильно: num_workers 0)"
                )
                nw = _cap
    if _google_colab_runtime():
        cw = os.environ.get("HTR_COLAB_NUM_WORKERS", "").strip()
        if cw != "":
            try:
                return max(0, int(cw))
            except ValueError:
                pass
        colab_cap = _colab_default_dataloader_workers_cap()
        if nw > colab_cap:
            print(
                f"[htr-train] Google Colab: num_workers {nw} → {colab_cap} "
                f"(адаптивно по cpu_count≈{os.cpu_count()}, без перегрузки RAM; свой предел: "
                "export HTR_COLAB_NUM_WORKERS=N; отключить воркеры: =0)"
            )
            nw = colab_cap
    return nw


def _preprocessed_ram_budget(dc: dict, nw: int) -> tuple[int, float]:
    raw = dc.get("preprocessed_ram_cache_max_gb", 6.0)
    if raw is None:
        gb = 6.0
    else:
        try:
            gb = float(raw)
        except (TypeError, ValueError):
            gb = 6.0
    if gb <= 0:
        return 0, gb
    total = gb * (1024.0**3)
    if nw <= 0:
        return max(int(total), 1024), gb
    denom = max(1, nw) * 2
    return max(int(total / denom), 1024), gb


def _training_source_entries(dc: dict) -> list[dict]:
    raw = dc.get("sources")

    def _parse_one(it: dict, idx: int) -> dict:
        if not isinstance(it, dict):
            raise TypeError(f"data.sources[{idx}] должен быть объектом YAML")
        k = str(it.get("kind") or "").strip().lower()
        if not k:
            if it.get("coco_json") and it.get("image_root"):
                k = "coco_lines"
            elif it.get("pair_root"):
                k = "page_txt_pairs"
            else:
                raise ValueError(
                    f"data.sources[{idx}]: укажите kind или (coco_json+image_root) или pair_root для page_txt_pairs"
                )
        ns = str(it.get("cache_namespace", "") or "").strip()
        if k == "coco_lines":
            cj, ir = it.get("coco_json"), it.get("image_root")
            if not cj or not ir:
                raise ValueError(f"data.sources[{idx}] (coco_lines): нужны coco_json и image_root")
            tf = it.get("text_field") or dc.get("text_field", "translation")
            return {"kind": "coco_lines", "coco_json": cj, "image_root": ir, "text_field": tf, "cache_namespace": ns}
        if k == "page_txt_pairs":
            pr = it.get("pair_root")
            if not pr:
                raise ValueError(f"data.sources[{idx}] (page_txt_pairs): нужен pair_root")
            mt_raw = it.get("max_text_chars")
            if mt_raw is None:
                mt_raw = dc.get("page_txt_max_chars_default")
            mt: Optional[int] = None if mt_raw is None else int(mt_raw)
            return {
                "kind": "page_txt_pairs",
                "pair_root": pr,
                "cache_namespace": ns,
                "max_text_chars": mt,
                "optional": bool(it.get("optional", False)),
            }
        raise ValueError(f"Неизвестный data.sources[{idx}].kind={k!r} (coco_lines | page_txt_pairs)")

    if isinstance(raw, list):
        if len(raw) > 0:
            return [_parse_one(raw[i], i) for i in range(len(raw))]
    elif raw is not None:
        raise TypeError("data.sources должен быть списком объектов YAML или опущен (legacy coco_json)")
    cj, ir = dc.get("coco_json"), dc.get("image_root")
    if not cj or not ir:
        raise ValueError("В data задайте coco_json + image_root или непустой список data.sources[]")
    return [
        {
            "kind": "coco_lines",
            "coco_json": cj,
            "image_root": ir,
            "text_field": dc.get("text_field", "translation"),
            "cache_namespace": "",
        }
    ]


def _disk_cache_slug(entry: dict) -> str:
    if entry.get("kind") == "page_txt_pairs":
        return hashlib.sha256(str(entry["pair_root"]).encode("utf-8")).hexdigest()[:12]
    return hashlib.sha256(str(entry["coco_json"]).encode("utf-8")).hexdigest()[:12]


def _disk_cache_root_for_source(dc: dict, entry: dict, n_sources: int) -> Optional[str]:
    raw = dc.get("preprocessed_cache_dir")
    if raw is None or not str(raw).strip():
        return None
    base = Path(str(raw).strip()).expanduser().resolve()
    ns = str(entry.get("cache_namespace", "") or "").strip()
    slug = _disk_cache_slug(entry)
    if n_sources <= 1:
        return str(base / ns) if ns else str(base)
    return str(base / (ns or slug))


def _try_set_multiprocessing_spawn_for_dataloader() -> None:
    """Linux по умолчанию fork — collate с .to(cuda) в воркере падает после CUDA в родителе. Spawn — отдельные процессы."""
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def _datasets_with_augment_backend(ds: torch.utils.data.Dataset) -> list[torch.utils.data.Dataset]:
    if isinstance(ds, ConcatDataset):
        return [x for x in ds.datasets if hasattr(x, "train_augment")]
    if hasattr(ds, "train_augment"):
        return [ds]
    return []


def run_training(cfg: dict) -> None:
    _try_set_multiprocessing_spawn_for_dataloader()
    objective = _training_objective(cfg)
    decoder_cap = _decoder_max_steps(cfg)
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    tc = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    val_max_batches = _optional_positive_int(tc.get("val_max_batches"))
    if val_max_batches is not None:
        print(f"[htr-train] val_max_batches={val_max_batches} (val-метрика только по первым N батчам — быстрее эпоха)")

    dc = cfg["data"]
    nw = _dataload_worker_count(dc)
    _hw_profile = hardware_profile(tc)
    wt_threads = effective_dataloader_worker_torch_threads(dc, _hw_profile, nw)
    apply_main_thread_env_and_torch(tc, nw=nw, wt_per_worker=max(1, wt_threads))
    if _hw_profile == "max":
        print(
            "[htr-train] hardware_utilization=max: усилен параллелизм CPU (main/interop threads, опционально OMP); "
            "prefetch_factor=auto и dataloader_worker_torch_threads=auto см. в конфиге/логах."
        )

    ram_budget_b, ram_gb_yaml = _preprocessed_ram_budget(dc, nw)

    train_aug = TrainAugmentation() if bool(dc.get("augmentation_train", False)) else None

    device_pref_early = str(cfg["project"].get("device", "cuda"))
    resolved_for_lines = pick_device(device_pref_early)
    lines_dev = torch.device(resolved_for_lines)
    use_gpu_line_pipe = (
        lines_dev.type in ("cuda", "mps")
        and bool(tc.get("cuda_line_resize_on_device", True))
        and train_aug is None
    )

    entries = _training_source_entries(dc)

    use_cuda_jpeg = (
        lines_dev.type == "cuda"
        and bool(tc.get("cuda_jpeg_decode_batched", True))
        and train_aug is None
        and use_gpu_line_pipe
    )
    if use_cuda_jpeg:
        print(
            "[htr-train] JPEG COCO: батч decode_jpeg(…, device=cuda) в collate; воркеры могут подавать только байты. "
            "PNG / page_txt_pairs — uint8 на CPU, финализация на GPU. Отключить: training.cuda_jpeg_decode_batched: false."
        )

    n_src = len(entries)
    ram_per_source = max(1024, ram_budget_b // max(1, n_src)) if ram_budget_b > 0 else 0

    parts: list[torch.utils.data.Dataset] = []
    kinds: list[str] = []
    for ent in entries:
        cr = _disk_cache_root_for_source(dc, ent, len(entries))
        k = str(ent["kind"])
        if k == "coco_lines":
            kinds.append(k)
            parts.append(
                COCOLinesDataset(
                    coco_json=ent["coco_json"],
                    image_root=ent["image_root"],
                    text_field=str(ent["text_field"]),
                    img_height=int(dc["img_height"]),
                    max_width=dc.get("max_width"),
                    min_crop_width=int(dc.get("min_crop_width", 4)),
                    train_augmentation=train_aug,
                    preprocessed_cache_dir=cr,
                    preprocessed_ram_cache_max_bytes=(ram_per_source if ram_budget_b > 0 else None),
                    cache_namespace=str(ent.get("cache_namespace", "") or ""),
                    defer_resize_normalize_to_cuda=use_gpu_line_pipe,
                    jpeg_decode_cuda_workers_zero=use_cuda_jpeg,
                )
            )
        elif k == "page_txt_pairs":
            opt = bool(ent.get("optional", False))
            try:
                parts.append(
                    PageTxtPairsDataset(
                        ent["pair_root"],
                        img_height=int(dc["img_height"]),
                        max_width=dc.get("max_width"),
                        train_augmentation=train_aug,
                        preprocessed_cache_dir=cr,
                        preprocessed_ram_cache_max_bytes=(ram_per_source if ram_budget_b > 0 else None),
                        cache_namespace=str(ent.get("cache_namespace", "") or ""),
                        max_text_chars=ent.get("max_text_chars"),
                        defer_resize_normalize_to_cuda=use_gpu_line_pipe,
                        defer_png_bytes_to_collate=use_cuda_jpeg,
                    )
                )
                kinds.append(k)
            except FileNotFoundError as ex:
                if opt:
                    print(f"[htr-train] optional источник page_txt_pairs пропускается: {ex}")
                    continue
                raise
        else:
            raise RuntimeError(f"внутренняя ошибка: неизвестный kind={k!r}")

    if not parts:
        raise RuntimeError(
            "Не загружен ни один источник данных (проверьте пути к COCO/json и парам страницы; "
            "для ROO см. unzip pages-img-plaintext; источники page_txt_pairs с optional: true можно пропустить)."
        )

    counts = [len(p) for p in parts]
    if len(parts) > 1:
        print(f"[htr-train] данные: {len(parts)} активных источников {kinds}; размеры поднаборов={counts} (итого {sum(counts)})")
    else:
        print(f"[htr-train] один источник ({kinds[0]}), размер={counts[0]}")

    if lines_dev.type in ("cuda", "mps") and use_gpu_line_pipe:
        print(
            "[htr-train] CPU/GPU: устройство — forward, AMP, float-линия; COCO JPEG — decode_jpeg(CUDA); "
            "COCO PNG / page_txt PNG — decode_png (ядро libpng CPU) минимизирует работу в воркере, "
            "серый/resize линии на CUDA; кэш u8 по-прежнему до байтов. Полностью GPU-декод PNG в PyTorch/torchvision без nvJPEG для PNG нет."
        )

    full_ds: torch.utils.data.Dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    _log_disk_cache_readiness(parts, kinds, counts)
    if ram_budget_b > 0:
        split_d = max(1, nw) * (2 if nw > 0 else 1)
        print(
            f"[htr-train] preprocessed_ram_cache_max_gb(total)≈{ram_gb_yaml:g} · "
            f"~{ram_budget_b / (1024**3):.3f} GiB per DataLoader worker "
            f"(split /{split_d}: train + val loaders × num_workers; num_workers={nw})"
            f"{f'; ~{ram_per_source/(1024**3):.3f} GiB LRU на каждый из {len(entries)} заявленных поднаборов в YAML' if len(entries) > 1 else ''}"
        )

    vf = float(dc.get("val_fraction", 0.0))
    train_ix, val_ix = random_split_indices(len(full_ds), vf, seed=seed)

    if len(train_ix) > 50_000:
        print(
            "[htr-train] сбор алфавита: читаем только подписи из разметки (без JPEG/PNG — иначе старт занимал бы часы)…"
        )
    texts_train = _collect_charset_texts(full_ds, train_ix, tc)

    extra = cfg.get("charset", {}).get("extra_chars") or ""
    charset = charset_from_strings(texts_train, extra)

    train_ds = Subset(full_ds, train_ix)
    backends = _datasets_with_augment_backend(full_ds)
    _bak_aug = [b.train_augment for b in backends]
    for b in backends:
        b.train_augment = None
    val_ds = Subset(full_ds, val_ix)
    for b, ag in zip(backends, _bak_aug):
        b.train_augment = ag

    from torch.utils.data import DataLoader

    bs = int(cfg["training"]["batch_size"])

    win_fast = _windows_dataloader_fast(dc) if sys.platform == "win32" else False
    colab_rt = _google_colab_runtime()
    if colab_rt and bs >= 768:
        print(
            "[htr-train] Colab: большой training.batch_size — узкое место чаще CPU/диск (JPEG + кэш .pt), не GPU. "
            "Часто быстрее по wall-clock: второй конфиг `configs/colab.yaml` (меньший batch, lr) или уменьшите batch_size."
        )

    worker_init_fn = _DataloaderWorkerTorchThreadsInit(wt_threads) if nw > 0 and wt_threads > 0 else None

    if win_fast and nw > 0:
        print(
            "[htr-train] Windows fast DataLoader: windows_dataloader_fast / HTR_WIN_DATALOADER_FAST=1 "
            "(до 8 workers, prefetch до 8, persistent_workers; возможен повтор 1455 — тогда выключите или num_workers 0)."
        )

    if use_cuda_jpeg:
        _collate = functools.partial(
            collate_gpu_lines_jpeg_cuda_batch,
            device=lines_dev,
            img_height=int(dc["img_height"]),
            max_width=dc.get("max_width"),
            min_crop_width=int(dc.get("min_crop_width", 4)),
        )
    elif use_gpu_line_pipe:
        _collate = coco_collate_mixed_lines
    else:
        _collate = coco_collate_fn

    # collate_gpu_lines_jpeg_cuda_batch кладёт image на CUDA — pin_memory недопустим (dense CPU only).
    _pin_memory = bool(torch.cuda.is_available() and not use_cuda_jpeg)
    _dl_common: dict = {
        "num_workers": nw,
        "collate_fn": _collate,
        "pin_memory": _pin_memory,
    }
    if worker_init_fn is not None:
        _dl_common["worker_init_fn"] = worker_init_fn
    if nw > 0:
        if colab_rt:
            pass  # без persistent_workers (ниже по умолчанию False для Colab)
        elif sys.platform != "win32":
            _dl_common["persistent_workers"] = True
        elif win_fast:
            _dl_common["persistent_workers"] = True
        try:
            _pf = effective_prefetch_factor(dc, nw, colab_rt=colab_rt, win_fast=win_fast)
        except Exception:
            _pf = 4
        _dl_common["prefetch_factor"] = _pf

    loader_train = DataLoader(train_ds, shuffle=True, batch_size=bs, **_dl_common)
    loader_val = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=max(1, bs // 2),
        **_dl_common,
    )

    train_loop_outer: object = loader_train
    if bool(tc.get("async_training_batch_prefetch", False)) and lines_dev.type in ("cuda", "mps"):
        qsz = max(1, min(8, int(tc.get("async_prefetch_queue_size", 2))))
        train_loop_outer = AsyncCpuBatchPrefetcher(loader_train, max_queue=qsz)
        print(
            f"[htr-train] async_training_batch_prefetch: до {qsz} CPU-батчей в фоне, пока GPU считает предыдущий"
        )

    device_pref = device_pref_early
    resolved = resolved_for_lines
    device = lines_dev

    if nw > 0:
        if colab_rt:
            _persist = False
        else:
            _persist = (sys.platform != "win32") or win_fast
        print(
            f"[htr-train] dataloader workers={nw} prefetch_factor={_dl_common.get('prefetch_factor')} "
            f"persistent_workers={_persist} pin_memory={_pin_memory} dataloader_worker_torch_threads="
            f"{wt_threads if wt_threads > 0 else 'off'}"
        )
        if device.type == "cuda" and worker_init_fn is not None and not use_cuda_jpeg:
            hint = (
                "[htr-train] подсказка: высокий CPU и низкая загрузка GPU при декоде JPEG — норма, "
                "особенно пока не заполнился preprocessed_cache_dir (первая эпоха дольше, дальше обычно быстрее). "
                "Усилить подачу: больше data.num_workers под ваш CPU, SSD под кэш, при необходимости batch_size↑."
            )
            if colab_rt:
                hint += " На Colab: export HTR_COLAB_NUM_WORKERS=6 (или ваш лимит) если ещё медленно."
            print(hint)

    if device.type == "cuda":
        if tc.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if tc.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(
        f"[htr-train] device: requested={device_pref!r} -> {device!r} "
        f"(torch.cuda.is_available={torch.cuda.is_available()})"
    )
    if use_gpu_line_pipe:
        if use_cuda_jpeg:
            print(
                "[htr-train] линии: промах u8-кэша — JPEG decode_jpeg(CUDA); PNG — decode_png→CUDA, ресайз/нормализация на GPU."
            )
            print(
                "[htr-train] VRAM: collate в воркерах использует CUDA — память делит несколько процессов с обучением. "
                "При OutOfMemory: уменьшите training.batch_size, data.num_workers, async_prefetch_queue_size; "
                "не запускайте второй эксперимент на той же GPU; при фрагментации: "
                "PYTORCH_ALLOC_CONF=expandable_segments:True"
            )
        else:
            print(
                "[htr-train] линии: float-линия на GPU после кропа; "
                "JPEG decode на GPU выключен (training.cuda_jpeg_decode_batched: false)."
            )
    if device_pref == "cuda" and not torch.cuda.is_available():
        _extra = ""
        if _google_colab_runtime():
            _extra = (
                " В Colab: Среда выполнения → Изменить тип среды → выберите GPU (например T4), затем Перезапустить сеанс;"
                " в ячейке: import torch; assert torch.cuda.is_available(); !nvidia-smi."
            )
        print(
            "[htr-train] WARNING: CUDA requested but not available; training on CPU. "
            "Install GPU build: https://pytorch.org/get-started/locally/"
            + _extra
        )
    model = resolve_model(cfg, charset.num_classes).to(device)
    model = _maybe_torch_compile(model, tc, device)

    freeze_ep = _freeze_backbone_epochs(cfg)

    criterion = nn.CTCLoss(blank=Charset.blank_idx, zero_infinity=True) if objective != "attention_ce" else None
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))

    if freeze_ep > 0:
        _set_pretrained_backbone_frozen(model, True)

    optim_ = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    amp_cfg = bool(cfg["training"].get("amp", False))
    scaler = _make_grad_scaler(enabled=(amp_cfg and device.type == "cuda"))

    epochs = int(cfg["training"]["epochs"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    experiment = cfg["training"].get("experiment_name", "run")

    for epoch in range(1, epochs + 1):
        if freeze_ep > 0 and isinstance(model, PretrainedResnetLineCTC):
            _set_pretrained_backbone_frozen(model, epoch <= freeze_ep)

        model.train()
        total_loss = 0.0
        nb_tr = 0
        train_bar = tqdm(train_loop_outer, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in train_bar:
            b_t = move_training_image_batch(
                batch,
                device,
                img_height=int(dc["img_height"]),
                max_width=dc.get("max_width"),
            )
            images = b_t["image"]  # type: ignore[arg-type]
            texts_batch: list[str] = b_t["text"]  # type: ignore[list-item]

            optim_.zero_grad(set_to_none=True)
            use_amp = scaler.is_enabled()

            if objective == "attention_ce":
                if not isinstance(model, AttentionLineSeq2Seq):
                    raise TypeError("objective attention_ce требует model.name attention_line_seq2seq")
                tin_y, targ_y, m_ok = _prepare_attention_batches(model, charset, texts_batch, device, decoder_cap)
                if use_amp:
                    with _autocast_cuda():
                        loss = model.compute_loss_ce(images, tin_y, targ_y, m_ok)
                    scaler.scale(loss).backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        scaler.unscale_(optim_)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optim_)
                    scaler.update()
                else:
                    loss = model.compute_loss_ce(images, tin_y, targ_y, m_ok)
                    loss.backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optim_.step()

            else:
                assert criterion is not None
                targets_tl, tgt_lengths = _pack_targets(texts_batch, charset, device)
                if use_amp:
                    with _autocast_cuda():
                        log_probs = model(images)
                        inp_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                        batch_loss = criterion(log_probs, targets_tl, inp_len, tgt_lengths)
                    scaler.scale(batch_loss).backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        scaler.unscale_(optim_)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optim_)
                    scaler.update()
                else:
                    log_probs = model(images)
                    inp_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                    batch_loss = criterion(log_probs, targets_tl, inp_len, tgt_lengths)
                    batch_loss.backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optim_.step()

                loss = batch_loss

            total_loss += float(loss.item())
            nb_tr += 1
            train_bar.set_postfix(loss=loss.item())

        mean_train = total_loss / max(1, nb_tr)
        cer_sum = 0.0
        n_lab = 0

        model.eval()
        with torch.no_grad():
            if not val_ix:
                print(f"[epoch {epoch}] train_loss={mean_train:.4f} (val: пусто val_fraction)")
            else:
                for vb_i, vbatch in enumerate(tqdm(loader_val, desc=f"val {epoch}", leave=False)):
                    if val_max_batches is not None and vb_i >= val_max_batches:
                        break
                    b_v = move_training_image_batch(
                        vbatch,
                        device,
                        img_height=int(dc["img_height"]),
                        max_width=dc.get("max_width"),
                    )
                    imgs_b = b_v["image"]  # type: ignore[arg-type]
                    refs_txt: list[str] = b_v["text"]  # type: ignore[list-item]
                    if objective == "attention_ce" and isinstance(model, AttentionLineSeq2Seq):
                        ce_preds = model.greedy_inference(imgs_b, decoder_cap)
                        for bi, sq in enumerate(ce_preds):
                            hyp_txt = _attention_hypothesis(sq, charset)
                            _, ratio_val = lev_ratio(hyp_txt, refs_txt[bi])
                            cer_sum += ratio_val
                            n_lab += 1
                    else:
                        log_p = model(imgs_b)
                        greedy_sequences = log_p.argmax(dim=-1).transpose(0, 1).cpu().tolist()
                        for bi, sq in enumerate(greedy_sequences):
                            hyp_txt = charset.decode_indices(sq)
                            _, ratio_val = lev_ratio(hyp_txt, refs_txt[bi])
                            cer_sum += ratio_val
                            n_lab += 1
                avg_cer = cer_sum / max(1, n_lab)
                val_note = (
                    f" [первые ≤{val_max_batches} val-батчей]"
                    if val_max_batches is not None
                    else ""
                )
                print(f"[epoch {epoch}] train_loss={mean_train:.4f} val_sym_error_ratio={avg_cer:.4f}{val_note}")

        save_every = max(1, int(cfg["training"].get("save_every_epochs", 1)))
        if epoch % save_every == 0:
            epath = ckpt_dir / f"{experiment}_e{epoch}.pt"
            save_checkpoint(
                str(epath),
                model.state_dict(),
                itos=charset.itos,
                model_name=str(cfg["model"]["name"]),
                yaml_dump=dict(cfg),
            )

    latest_ck = ckpt_dir / "latest.pt"
    save_checkpoint(
        str(latest_ck),
        model.state_dict(),
        itos=charset.itos,
        model_name=str(cfg["model"]["name"]),
        yaml_dump=dict(cfg),
    )
