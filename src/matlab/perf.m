function [wb] = perf(x, t)
    global wbsys wbt

    if ~wbt
        wb = wbsys;
    else
        wb = wbsys * (1 + 2 * t + t^2);
    end


