"""変更(v7): nvccなしでも.cu内の計算式をCPUエミュレーションして検証する。

CUDAランタイム、起動構文、PyTorch C++連携の検証は実機テストが担当する。
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/cuda/simulation_v7.cu'


@unittest.skipUnless(shutil.which('c++'), 'C++ compiler is unavailable')
class NativeKernelLogicTests(unittest.TestCase):
    def test_actual_cuda_kernel_formulas(self):
        source = SOURCE.read_text()
        # カーネル本体を再実装せず、そのまま通常C++として実行する。
        kernels = source[source.index('__device__ I wrap'):source.index('T sorted_keys')]
        harness = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
using I = int64_t;
#define __device__
#define __global__
struct Dim { int x; } blockIdx{0}, blockDim{1}, threadIdx{0};
void atomicAdd(unsigned long long* out, unsigned long long v) { *out += v; }
'''
        cases = r'''
int main() {
    // 1セル周期境界：自分を除外し、全員に到達可能な抽選順位を使う。
    I p3[6]={0,0,0,0,0,0}, food[1]={3}, cells[1]={0}, keys3[3]={0,1,2};
    I neighbor3[3], foods3[3];
    double random3[6]={0.,0.,.9,0.,.5,0.};
    for(int i=0;i<3;++i) { threadIdx.x=i;
        observe_kernel(p3,food,cells,keys3,random3,neighbor3,foods3,3,1,1); }
    assert(neighbor3[0]==1 && neighbor3[1]==2 && neighbor3[2]==1);
    for(int i=0;i<3;++i) assert(foods3[i]==0);
    // 多セルの近傍と自分の除外を、独立に列挙した候補と比較する。
    for(I size=1;size<=5;++size) {
        const I n=17;
        std::vector<I> pos(2*n), keys(n), grid(size*size,1), table;
        std::vector<I> offsets;
        for(I dx=-1;dx<=1;++dx) { I d=wrap(dx,size);
            if(std::find(offsets.begin(),offsets.end(),d)==offsets.end()) offsets.push_back(d); }
        for(I c=0;c<size*size;++c) for(I dx:offsets) for(I dy:offsets)
            table.push_back(wrap(c/size+dx,size)*size+wrap(c%size+dy,size));
        I k=offsets.size()*offsets.size();
        for(I i=0;i<n;++i) { pos[2*i]=(i*3)%size; pos[2*i+1]=(i*7+1)%size;
            keys[i]=(pos[2*i]*size+pos[2*i+1])*n+i; }
        std::sort(keys.begin(),keys.end());
        for(I self=0;self<n;++self) {
            I cell=pos[2*self]*size+pos[2*self+1]; std::vector<I> candidates;
            for(I j=0;j<k;++j) for(I key:keys)
                if(key/n==table[cell*k+j] && key%n!=self) candidates.push_back(key%n);
            for(I rank=0;rank<static_cast<I>(candidates.size());++rank) {
                std::vector<double> u(2*n,0.); std::vector<I> neighbors(n), foods(n);
                u[2*self]=(rank+.5)/candidates.size(); threadIdx.x=self;
                observe_kernel(pos.data(),grid.data(),table.data(),keys.data(),u.data(),
                               neighbors.data(),foods.data(),n,size,k);
                assert(neighbors[self]==candidates[rank]);
            }
        }
    }
    // 行動・移譲は更新前の状態を参照し、受取後に死亡判定する。
    I pos[4]={0,0,0,0}, weights[16]={0}; weights[5]=1;
    double energies[2]={10.,-50.}; I neighbors[2]={1,0}, foods[2]={-1,-1}, dirs[4]={0};
    I moved[4]; double transfers[2];
    for(int i=0;i<2;++i) { threadIdx.x=i;
        compute_kernel(pos,energies,weights,neighbors,foods,dirs,moved,transfers,
                       2,1,-45.,0.,-50.,3.); }
    assert(transfers[0]==3. && transfers[1]==0.);
    I rkeys[2]; for(int i=0;i<2;++i) { threadIdx.x=i; recipient_keys_kernel(neighbors,rkeys,2); }
    std::sort(rkeys,rkeys+2);
    double rewards[2]={0.,0.}, next[2]; I ages[2]={4,8}, repeats[2], dead[2], altruism=0;
    for(int i=0;i<2;++i) { threadIdx.x=i;
        energy_kernel(energies,ages,rkeys,transfers,rewards,next,repeats,dead,&altruism,2,0.,-50.,100.); }
    assert(next[0]==7. && next[1]==-47. && altruism==1 && !dead[1]);
    // 餌競合：priority=[2,0,1]に対して先頭2名だけが報酬を得る。
    I sorted[3]={0,1,2}, inverse[3]={1,2,0}, limited[1]={2}, consumed[1]; double reward[3];
    for(int i=0;i<3;++i) { threadIdx.x=i; food_kernel(sorted,inverse,limited,reward,3,2.); }
    assert(reward[0]==0. && reward[1]==2. && reward[2]==2.);
    threadIdx.x=0; consumed_kernel(sorted,limited,consumed,3,1); assert(consumed[0]==2);
    // 死亡個体は除外し、繁殖親の直後に変異した子を配置する。
    I cp[6]={0,0,1,1,2,2}, ca[3]={4,8,12}, cw[24]={0}, cr[3]={2,0,1};
    I prefix[3]={2,2,3}, cd[3]={0,1,0}, deathprefix[3]={0,1,1};
    double ce[3]={100.,-50.,3.}, u[24]={0};
    I op[6], oa[3], ow[24], deathage[1]; double oe[3];
    for(int i=0;i<3;++i) { threadIdx.x=i;
        commit_kernel(cp,ce,ca,cw,cr,prefix,cd,deathprefix,u,op,oe,oa,ow,deathage,3,1.); }
    assert(oa[0]==5 && oa[1]==0 && oa[2]==13 && deathage[0]==9);
    assert(oe[0]==0. && oe[1]==0. && oe[2]==3.);
    for(int k=0;k<8;++k) assert(ow[k]==0 && ow[8+k]==1 && ow[16+k]==0);
    // 移動の4行動と、偶数長トーラスの半周方向。
    I mp[2]={0,0}, mw[8]={0}, mn[1]={-1}, mf[1]={2*4}, md[2]={0};
    double me[1]={0.}, mt[1]; I mo[2];
    for(I a0=0;a0<2;++a0) for(I a1=0;a1<2;++a1) {
        mw[0]=a0; mw[4]=a1; threadIdx.x=0;
        compute_kernel(mp,me,mw,mn,mf,md,mo,mt,1,4,-45.,.1,-50.,3.);
        assert(mo[0]==(a1 ? 0 : (a0 ? 1 : 3)) && mo[1]==0);
    }
    // 再生成は消費後に実施。空の個体群でも餌生成は継続する。
    double spawn_random[1]={0.}; I f[1]={2}; threadIdx.x=0;
    spawn_kernel(f,consumed,spawn_random,1,1.,5); assert(f[0]==5);
    spawn_kernel(f,consumed,spawn_random,1,0.,5); assert(f[0]==3);
    consumed_kernel(nullptr,f,consumed,0,1); assert(consumed[0]==0);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / 'kernels.cpp'
            exe = Path(directory) / 'kernels'
            cpp.write_text(harness + kernels + cases)
            compile_result = subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                                              str(cpp), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
