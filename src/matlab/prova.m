function [sol] = prova

        x = linspace(0,1,101);
        t = linspace(0,1,101);
        [~, ~, y1, y2, y3] = ic_bc(0, t);
        disp(y2)
