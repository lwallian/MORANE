function [block] = nextMatrixPeriod(m, blockSize,currPosition, mode)
%NEXTMATRIXPERIOD Gets the next diagonals given a count and the current
%position. If the position is the end of the matrix or bigger, a cell with
%a 0 in its first component is returned. Otherwise, if the size is too big,
%the rest of the diagonals is returned.
%   @param m: the whole matrix
%   @param blockSize: the dimension of the block (amount of diagonals)
%   @param currPosition: the current column index
%   @param mode: puts the actual diagonal in first place if 'global'
%   @return: cell with the requested diagonals
%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%
[N, M] = size(m);
minDimension = min(N, M); % I'm not sure that it would work in all cases

assert(strcmp(mode, 'global') || strcmp(mode, 'local'));

if currPosition >= minDimension
    block{1} = 0;
    return
elseif minDimension - currPosition < blockSize
    blockSize = minDimension - currPosition;
end

if strcmp(mode, 'local')
    block = cell(blockSize, 1);
    
    for i = 1 : blockSize
        for j = 1 : minDimension - currPosition - i + 1
            block{i}(j) = m(j, j + i + currPosition - 1);
        end
    end
else
    block = cell(blockSize + 1, 1);
    
    block{1} = diag(m);
    for i = 1 : blockSize
        for j = 1 : minDimension - currPosition - i + 1
            block{i + 1}(j) = m(j, j + i + currPosition - 1);
        end
    end
end

end
