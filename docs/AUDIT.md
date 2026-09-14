# Phase 0 audit

## BoltzGen
```text
BOLTZGEN_ROOT=/share/home/rongdingyi/programs/proteingen/boltzgen
https://github.com/HannesStark/boltzgen
a3149cf18eeb58648d1abbb27539bd73f746cdda
checkpoint=/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_ifold.ckpt
checkpoint_sha256=dd4cf108c94471bdc3a326b7b180fa3854dc019110fae780208c30b50bd56578
python=Python 3.9.18
```

### IF anatomy (grep evidence)
```text
517:class InverseFoldingDecoder(nn.Module):
268:    def sample(self, s, z):
647:    def sample(self, s, z, edge_idx, valid_mask, feats):
586:        design_mask: Tensor,
599:            if design_mask[i]:
651:        if "inverse_fold_design_mask" in feats:
652:            design_mask = feats["inverse_fold_design_mask"].bool()[valid_mask]
654:            design_mask = feats["design_mask"].bool()[valid_mask]
655:        num_not_design = (~design_mask).sum().item()
748:                ids_canonical = torch.multinomial(
536:        sampling_temperature: float = 0.1,
558:        self.sampling_temperature = sampling_temperature
745:            if self.sampling_temperature is None:
749:                    F.softmax(pred_canonical / self.sampling_temperature, dim=-1),
375:            self.structure_module = InverseFoldingDecoder(**inverse_fold_args)
528:                edge_idx, valid_mask, s, z = self.inverse_folding_encoder(feats)
```

## VHH scorer
```text
VHH_SCORER_ROOT=/share/home/rongdingyi/programs/proteingen/vhh_guidance
30:    camelid_native_likeness_score: float
31:    nativeness_margin: float
394:        nativeness_margin = torch.stack(
431:            "nativeness_margin": nativeness_margin,
439:    def score_sequences(
```
