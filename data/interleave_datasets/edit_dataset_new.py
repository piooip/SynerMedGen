# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        """
        支持三种输入形式（向后兼容）：
        1) 经典链式：image_list=[src1, src2, ..., tgt]，instruction_list 长度 = len(image_list)-1
           -> 原逻辑，随机 [start_idx, end_idx]，逐步或合并训练
        2) 合并单指令（新增）：image_list=[src1, src2, ..., tgt]，instruction_list 仅 1 条（变体列表）
           -> 编码前 N-1 张为上下文，喂这一条总指令，监督第 N 张
        3) 明确 combined_instruction 字段（新增）：row["combined_instruction"] 为 str/list，image_list 同上
           -> 与(2)同处理
        """
        def _choose_one_text(x):
            # 支持 str / list[str] / list[list[str]] 的宽松输入
            if x is None:
                return None
            if isinstance(x, str):
                return x
            if isinstance(x, list):
                if not x:
                    return None
                # 如果是 [["a","b","c"]] 这种，取第一组再随机；如果是 ["a","b","c"]，直接随机
                if isinstance(x[0], list):
                    return random.choice(x[0]) if x[0] else None
                return random.choice(x)
            return None

        image_num = len(row["image_list"])
        assert image_num >= 1, "row['image_list'] must have at least 1 image."

        # ===== Combined mode：多图 + 单条总指令 =====
        combined_text = None
        # 优先读取显式字段
        if "combined_instruction" in row:
            combined_text = _choose_one_text(row["combined_instruction"])
        # 若未提供显式字段，则当 instruction_list 仅 1 条时触发
        if combined_text is None and "instruction_list" in row:
            try:
                if image_num >= 2 and len(row["instruction_list"]) == 1:
                    combined_text = _choose_one_text(row["instruction_list"][0])
            except Exception:
                combined_text = None  # 不阻断后续老路径

        if combined_text is not None and image_num >= 2:
            # -------- Combined mode 路径：前 N-1 张作为上下文，最后一张为监督目标 --------
            data = self._init_data()

            # 编码所有“源图”（前 N-1 张）：进入缓存，不计损
            for i in range(image_num - 1):
                img = pil_img2rgb(Image.open(io.BytesIO(row["image_list"][i])))
                data = self._add_image(
                    data,
                    img,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )

            # 单条总指令文本（不计损）
            data = self._add_text(data, combined_text, need_loss=False)

            # 最后一张作为监督目标：不再过 VAE/ViT，直接计算 loss
            tgt_img = pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-1])))
            data = self._add_image(
                data,
                tgt_img,
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
            return data

        # ===== 否则走原有逻辑：链式逐步/合并（保持完全兼容） =====
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )

        if end_idx - start_idx > 1 and random.random() < 0.5:  # concat multiple instruction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx - 1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx - 1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=True,
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )
        return data
