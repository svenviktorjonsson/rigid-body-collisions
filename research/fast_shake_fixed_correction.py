"""Prospectively declared equivalent fixed-step controls after input rejection."""
import argparse
from research import fast_shake_extension as extension

def aligned_errors(engine,left,right):
    common=sorted(set(round(t,12) for t in left['times']) & set(round(t,12) for t in right['times']))
    def align(result):
        indices={round(t,12):i for i,t in enumerate(result['times'])}
        return dict(result,times=common,states=[result['states'][indices[t]] for t in common])
    return engine.errors(align(left),align(right))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    extension.PLAN=extension.DIRECTORY/'fixed-correction-plan.json'
    if args.summarize:
        factory=extension.adapter
        def with_common_samples():
            original=factory()
            # Module objects cannot be copied: expose only the operation used
            # by the evidence summarizer through a small explicit wrapper.
            class Proxy:
                errors=staticmethod(lambda left,right:aligned_errors(original,left,right))
            return Proxy()
        extension.adapter=with_common_samples
        existing={name:(extension.DIRECTORY/name).read_bytes() for name in ('extension-summary.json','extension-traces.zip') if (extension.DIRECTORY/name).exists()}
        extension.summarize()
        (extension.DIRECTORY/'extension-summary.json').replace(extension.DIRECTORY/'corrected-extension-summary.json')
        (extension.DIRECTORY/'extension-traces.zip').replace(extension.DIRECTORY/'corrected-extension-traces.zip')
        for name,value in existing.items():(extension.DIRECTORY/name).write_bytes(value)
    else:extension.run_lane('fixed_1_25us_corrected')

if __name__=='__main__':main()
