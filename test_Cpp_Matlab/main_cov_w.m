function main_cov_w()
bt = zeros(1000,8);

for i = 1:1000
    for j = 1:8
        bt(i,j) = log(i*j);
    end
end

cov_w = bt * bt';
n = norm(cov_w);

end

