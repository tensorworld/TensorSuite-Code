#!/usr/bin/env python3

# This file is part of ParTI!.
#
# ParTI! is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# ParTI! is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with ParTI!.
# If not, see <http://www.gnu.org/licenses/>.


# Usage: python3 random_tensor_generator.py <output> <n%dimSize> <n%dimSize> <n%dimSize>....
# explaination: 
#   <output>: output file and path
#   n%:density of the corresponding dimension
#   dimSize:corrosponding dimension size
#
# value~N(0,1)
# 
# Output
# NNZ, memory size
# Example: "16807 generated, 337.2 KiB written.""

# ToDo:
# add a new input params "precision", precision=D:double float store the value; precision=S: single precision
# add a new input params "MTH(Memory Threshold)": the limited memory boundary(GB)
# after receive the params, first calculate the estimate memory cost, if > threshold, return, print("")
# There's a viarance between estmate NNZ and real NNZ, estimation may be bigger/smaller. In the following implementation ,we ingnore this issue. simply estimate NNZ*NNZ_size<threshold
# 
import math
import random
import sys

if sys.version_info < (3,):
    range = xrange


def randround(x):
    int_part = math.floor(x)
    frac_part = x - int_part
    return int(math.ceil(x) if random.random() < frac_part else int_part)


def human_size(nbytes):
    if nbytes < 1024:
        return '%d bytes' % nbytes
    elif nbytes < 1048576:
        return '%.1f KiB' % (nbytes / 1024.0)
    elif nbytes < 1073741824:
        return '%.1f MiB' % (nbytes / 1048576.0)
    else:
        return '%.2f GiB' % (nbytes / 1073741824.0)


def main(argv):
    if len(argv) < 5:
        print('Usage:   %s output.tns <precision:S/D> <MTH_GB> [nonzero_rate%%dim_size] ...' % argv[0])
        print('Example: %s output.tns D 2.0 50%%1024 2%%4096' % argv[0])
        print()
        print('precision: S(single,4B) or D(double,8B)')
        print('MTH_GB: memory threshold in GB')
        print('Each non-zero element ~ N(0,1)')
        print()
        return 1

    output = argv[1]
    precision = argv[2].upper()
    threshold_gb = float(argv[3])
    rates, dims = [], []

    for i in argv[4:]:
        if '%' in i:
            rate, dim = i.split('%', 1)
            rates.append(float(rate) * 0.01)
            dims.append(int(dim))
        else:
            rates.append(1)
            dims.append(int(i))

    ndims = len(dims)

    # ----------------------------
    # Estimate NNZ and memory cost
    # ----------------------------
    nnz_est = 1.0
    for i in range(ndims):
        nnz_est *= rates[i] * dims[i]
    nnz_est = round(nnz_est)

    # COO format: each entry = ndims indices (4B each) + value (4B or 8B)
    index_bytes = ndims * 4
    val_bytes = 8 if precision == 'D' else 4
    entry_bytes = index_bytes + val_bytes
    est_bytes = nnz_est * entry_bytes
    est_gb = est_bytes / (1024 ** 3)

    print('--------------------------------------------')
    print('Estimated NNZ:          %d' % nnz_est)
    print('Estimated Memory usage: %s (%.3f GiB)' % (human_size(est_bytes), est_gb))
    print('Precision:              %s (%d bytes per value)' % (precision, val_bytes))
    print('Memory Threshold:       %.3f GiB' % threshold_gb)
    print('--------------------------------------------')

    if est_gb > threshold_gb:
        print('❌ Estimated tensor size exceeds threshold. Generation aborted.')
        return 0

    print('✅ Under threshold, start generating tensor...')
    written = 0
    percent = 0

    f = open(output, 'w')
    f.write('%d\n' % ndims)
    f.write('\t'.join(map(str, dims)))
    f.write('\n')

    inds = [None] * ndims
    ptrs = [0] * ndims
    for i in range(ndims):
        if rates[i] == 1:
            inds[i] = range(dims[i])
        else:
            inds[i] = random.sample(range(dims[i]), randround(rates[i] * dims[i]))
            inds[i].sort()

    while ptrs[0] != len(inds[0]):
        for i in range(ndims):
            f.write('%d\t' % (inds[i][ptrs[i]] + 1))

        # write value according to precision
        val = random.gauss(0, 1)
        if precision == 'D':
            f.write('% .16f\n' % val)
        else:
            f.write('% .6f\n' % val)

        ptrs[ndims - 1] += 1
        written += 1

        if nnz_est != 0:
            new_percent = int(written * 100.0 / nnz_est)
            if new_percent < 100 and new_percent != percent:
                percent = new_percent
                print('%3d%% completed, %d generated, %s written.' %
                      (percent, written, human_size(f.tell())), end='\r', flush=True)

        for i in range(ndims - 1, 0, -1):
            if ptrs[i] == len(inds[i]):
                if rates[i] == 1:
                    inds[i] = range(dims[i])
                else:
                    inds[i] = random.sample(range(dims[i]), randround(rates[i] * dims[i]))
                    inds[i].sort()
                ptrs[i] = 0
                ptrs[i - 1] += 1

    print('100%% completed, %d generated, %s written.' % (written, human_size(f.tell())))
    f.close()
    print('Successfully written into %s.' % output)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
