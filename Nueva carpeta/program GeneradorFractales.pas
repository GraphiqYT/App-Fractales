program GeneradorFractales;
uses graph, crt;

var
  gd, gm, tipo: integer;
  x, y, it: integer;
  cx, cy, zx, zy, tmpX, c_julia_r, c_julia_i: double;

begin
  writeln('Selecciona el fractal:');
  writeln('1. Mandelbrot');
  writeln('2. Conjunto de Julia (Clasico)');
  writeln('3. Burning Ship (Barco en llamas)');
  readln(tipo);

  gd := detect;
  initgraph(gd, gm, '');

  // Parametros para Julia
  c_julia_r := -0.7; c_julia_i := 0.27015;

  for y := 0 to getmaxy do
    for x := 0 to getmaxx do
    begin
      // Escalar coordenadas
      cx := (x - getmaxx / 2) * 4.0 / getmaxx;
      cy := (y - getmaxy / 2) * 4.0 / getmaxy;
      
      it := 0;
      case tipo of
        1: begin zx := 0; zy := 0; end; // Mandelbrot inicia en 0
        2: begin zx := cx; zy := cy; cx := c_julia_r; cy := c_julia_i; end; // Julia
        3: begin zx := 0; zy := 0; end; 
      end;

      while (zx*zx + zy*zy <= 4) and (it < 100) do
      begin
        case tipo of
          1, 2: tmpX := zx*zx - zy*zy + cx;
          3: tmpX := zx*zx - zy*zy + cx; // Lógica base
        end;

        if tipo = 3 then // Especial para Burning Ship
        begin
          zy := abs(2*zx*zy) + cy;
          zx := abs(tmpX); // El valor absoluto crea el efecto de "barco"
        end
        else
        begin
          zy := 2*zx*zy + cy;
          zx := tmpX;
        end;
        inc(it);
      end;

      if it < 100 then putpixel(x, y, it mod 16) else putpixel(x, y, black);
    end;

  readkey;
  closegraph;
end.
