"""Verify directed masking without loading the model weights or CUDA stack."""
import importlib.util
import math
from pathlib import Path
import unittest
from unittest import mock

import torch

spec=importlib.util.spec_from_file_location('native_attention',Path(__file__).parents[1]/'wan/modules/attention.py')
attention=importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


def dense(q,k,v,**kwargs):
    scores=torch.einsum('bqhd,bkhd->bqhk',q,k)/math.sqrt(q.size(-1))
    return torch.einsum('bqhk,bkhd->bqhd',scores.softmax(-1),v)


class PrefixMaskTest(unittest.TestCase):
    def inputs(self):
        gen=torch.Generator().manual_seed(48216)
        q,k,v=[torch.randn(2,12,2,8,generator=gen) for _ in range(3)]
        phases=torch.arange(16).float()[:,None]*torch.tensor([1.,.1,.01,.001])
        return dict(q=q,k=k,v=v,seq_lens=torch.tensor([12,8]),
                    grid_sizes=torch.tensor([[6,1,2],[4,1,2]]),anchor_index=3,
                    rope_compression=True,freqs=torch.polar(torch.ones_like(phases),phases))

    def test_all_prefix_queries_drop_only_anchor_and_suffix_is_identical(self):
        args=self.inputs()
        calls=[]
        def record(q,k,v,**kwargs):
            calls.append((q.clone(),k.clone(),v.clone()))
            return dense(q,k,v,**kwargs)
        with mock.patch.object(attention,'flash_attention',record):
            baseline=attention.joint_img2_guidance_attention(**args)
            baseline_calls=calls.copy();calls.clear()
            masked=attention.joint_img2_guidance_attention(**args,prefix_no_anchor=True)
        self.assertEqual(len(calls),4)
        self.assertEqual(calls[0][0].size(1),6)  # All three prefix frames.
        expected_k=torch.cat([baseline_calls[0][1][:,:6],baseline_calls[0][1][:,8:]],dim=1)
        expected_v=torch.cat([baseline_calls[0][2][:,:6],baseline_calls[0][2][:,8:]],dim=1)
        torch.testing.assert_close(calls[0][1],expected_k,rtol=0,atol=0)
        torch.testing.assert_close(calls[0][2],expected_v,rtol=0,atol=0)
        torch.testing.assert_close(masked[0:1,:6],dense(calls[0][0],expected_k,expected_v),rtol=0,atol=0)
        for actual,old in zip(calls[1:],baseline_calls[1:]):
            for value,previous in zip(actual,old):
                torch.testing.assert_close(value,previous,rtol=0,atol=0)
        torch.testing.assert_close(masked[0,6:],baseline[0,6:],rtol=0,atol=0)
        torch.testing.assert_close(masked[1],baseline[1],rtol=0,atol=0)
        self.assertFalse(torch.equal(masked[0,:6],baseline[0,:6]))

    def test_mask_requires_compression(self):
        args=self.inputs();args['rope_compression']=False
        with self.assertRaisesRegex(ValueError,'requires joint RoPE compression'):
            attention.joint_img2_guidance_attention(**args,prefix_no_anchor=True)


if __name__=='__main__':
    unittest.main()
