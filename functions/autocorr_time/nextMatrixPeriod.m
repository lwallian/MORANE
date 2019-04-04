function [block] = nextMatrixPeriod(m, blockSize,currPosition)
%GETMATRIXBLOCK Gets the next square matrix block partitioned from the
%diagonal in a symetric manner. If the block size is too big, a smaller
%matrix is returned. If the position is the end of the matrix, a 0 is
%returned.
%   Starting from the current position gets the next matrix block
%   @param m: the whole matrix
%   @param blockSize: the dimension of the block (blockSize x blockSize)
%   @param currPosition: the current index in the diagonal
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
[N, M] = size(m);
minDimension = min(N, M);

if minDimension == currPosition
    block = 0;
    return
elseif minDimension - currPosition < blockSize
    blockSize = minDimension - currPosition;
end

block = cell(blockSize, 1);

for i = 1 : blockSize
    for j = currPosition : N - currPosition - i + 1
        block{i}(j) = m(j, j + i - 1);
    end
end

end
