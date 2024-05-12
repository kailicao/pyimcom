($tag, $band, $dir) = @ARGV;

# original directory used for this script -- deprecated --> $dir = '/fs/scratch/PCON0003/cond0007/anl-run-out-tests';

## Generate data ##

chdir q:..:;
system "python -m diagnostics.dynrange $dir/$tag\_$band diagnostics/results/$tag-$band\_ > diagnostics/results/$tag-$band\_dynrange.dat";
system "python -m diagnostics.starcube_nonoise $band $dir/$tag\_ diagnostics/results/$tag-";
chdir 'diagnostics';

$N=0;
@lines = split "\n", `cat results/$tag-_StarCat_$band.txt`;
for $line (@lines) {
  if ($line !~ m/^\#/) {
    @data = split ' ', $line;
    for $i (0..21) {$alldata[$i][$N] = $data[$i];}
    $N++;
  }
}
print "$N stars\n";
# RMS ellipticity and size
($cm, $sizemed) = split ' ', `head -n 1 results/$tag-_StarCat_$band.txt`;
$evar = 0.;
$ewinvar = 0.;
$sizerr = 0.;
for $j (0..$N-1) {
  $evar += ($alldata[14][$j]**2+$alldata[15][$j]**2)/$N;
  $ewinvar += ($alldata[18][$j]**2+$alldata[19][$j]**2)/$N;
  $sizerrvar += ($alldata[13][$j]-$sizemed)**2/$N;
}

## Make plots ##

$tt = $tag;
$tt =~ s/\_/\\\\\\\_/mg;

$fname = "results/$tag-$band\_dynrange.dat";
@data_ = split "\n", `sed \'/\^\#/d\' $fname`;
($xmax) = (split ' ', $data_[-1])[0]; # max x-value
($ymax) = (split ' ', $data_[0])[-1]; # max y-value
($ymin) = (split ' ', $data_[-1])[2]; # max y-value
if ($ymin>=-30) {$ymin=-30;}
if ($ymax<50000) {$ymax=50000;}
open(G, "| gnuplot") or die;
print G qq~set term postscript enhanced eps 16 color\n~;
print G qq~set output "results/$tag-$band\_all.eps"\n~;
print G qq~set size 3,2.8\n~;
print G qq~set origin 0,0\n~;
print G qq~set multiplot\n~;
#
##########################
# dynamic range plot
##########################
#
print G qq~set size 1,1.4\n~;
print G qq~set origin 0,1.4\n~;
print G qq~set grid\n~;
print G qq~set xlabel "radius (s_{out})"\n~;
print G qq~set ylabel "intensity (e/s@^2_{in}/p)"\n~;
print G qq~set title "star profiles [case: ~.$tt.qq~ band: $band]"\n~;
print G qq~unset key\n~;
$os = 10; if (abs($ymin)*2>$os) {$os=abs($ymin)*2;}
print G qq~f(x)=log($os+x)\n~;
print G qq~set xrange [0:$xmax]; set xtics 4\n~;
print G qq~set yrange [f($ymin):f($ymax)]\n~;
print G qq~set ytics (~;
@r = (-100);
for $c (-9..1) {@r = (@r, $c*10);}
for $c (2..10) {@r = (@r, $c*10);}
for $c (2..10) {@r = (@r, $c*100);}
for $c (2..10) {@r = (@r, $c*1000);}
for $c (2..10) {@r = (@r, $c*10000);}
$q__ = '';
for $y (@r) {
  if ($y>=$ymin and $y<=$ymax) {
    if (abs($y)==10 or abs($y)==100 or abs($y)==1000 or abs($y)==10000 or $y==0) {
      $q__.= qq["$y" f($y),];
    } else {
      $q__.= qq["" f($y),];
    }
  }
}
chop($q__);
print G qq~$q__)\n~;
print G qq~set style line 1 lt 1 lc rgb "#000000" lw 3\n~;
print G qq~set style line 2 lt 1 lc rgb "#0040ff" lw 1\n~;
print G qq~set label "N=$N" at 40,f(20000)\n~;
print G qq~plot "$fname" using (\$1):(f(\$6)) with lines ls 1~;
for $icol (3..5,7..9) {
  print G qq~, "$fname" using (\$1):(f(\$$icol)) with lines ls 2 ~;
}
print G qq~\n~;
#
##########################
# ellipticity plots
##########################
#
print G qq~unset label\n~;
print G qq~set origin 1,0\n~;
print G qq~set xrange [30:70]\n~;
print G qq~set yrange [1e-5:1e-2]\n~;
print G qq~set style line 1 pt 7 ps 0.5 lc rgb "#008040"\n~;
print G qq~set style line 2 pt 65 ps 0.5 lc rgb "#e06000"\n~;
print G qq~set title "ellipticities [case: ~.$tt.qq~ band: $band]"\n~;
print G qq~set xlabel "Fidelity"\n~;
print G qq~set ylabel "|g_{out}|"\n~;
print G qq~unset key\n~;
print G qq~set grid\n~;
print G qq~set xtics 5\n~;
$q__ = '';
for $i (-5..-2) {
  $x = 10**$i;
  $q__.= qq[\"10\^\{$i\}\" $x,];
  for $c (2..9) {
    $x = 10**$i*$c;
    $q__.= qq["" $x,];
  }
}
chop($q__);
print G qq~set ytics ($q__); set logscale y\n~;
print G (sprintf qq~set label "rms = %11.5E" at 60,.008\n~, $evar**.5);
print G qq~plot "results/$tag-_StarCat_$band.txt" using (\$21):(sqrt(\$15**2+\$16**2)) with points ls 1\n~;
#
# windowed ellipticity
print G qq~unset label\n~;
print G qq~set origin 2,0\n~;
print G qq~set ylabel "|e_{out}| (0.4 arcsec window)"\n~;
print G (sprintf qq~set label "rms = %11.5E" at 60,.008\n~, $ewinvar**.5);
print G qq~plot "results/$tag-_StarCat_$band.txt" using (\$21):(sqrt(\$19**2+\$20**2)) with points ls 1\n~;
#
# size deviation from median
print G qq~unset label\n~;
print G qq~set origin 0,0\n~;
print G qq~set title "size errors [case: ~.$tt.qq~ band: $band]"\n~;
print G qq~set ylabel "{/Symbol s}-{/Symbol s}_{med}/{/Symbol s}_{med}"\n~;
print G (sprintf qq~set label "rms = %11.5E" at 58,.008\n~, $sizerrvar**.5/$sizemed);
print G (sprintf qq~set label "{/Symbol s}_{med} = %.5f s_{out}" at 58,.005\n~, $sizemed);
print G qq~set style line 1 pt 9 ps 0.5 lc rgb "#00a040"\n~;
print G qq~set style line 2 pt 11 ps 0.5 lc rgb "#e06000"\n~;
print G qq~plot "results/$tag-_StarCat_$band.txt" using (\$21):((\$14-$sizemed)/$sizemed) with points ls 1, ~;
print G qq~ "results/$tag-_StarCat_$band.txt" using (\$21):(-(\$14-$sizemed)/$sizemed) with points ls 2\n~;
#
##########################
# histograms
##########################
#
print G qq~unset label\n~;
print G qq~set origin 2,1.4\n~;
print G qq~set title "sqrtS histogram [case: ~.$tt.qq~ band: $band]"\n~;
print G qq~set xrange [0:2]; set xtics .2\n~;
($cs, $ymax, $pc) = split ' ', `head -n 1 results/$tag-$band\__sqrtS_hist.dat`;
print G qq~set yrange [0.9:1.1*$ymax]; set logscale y\n~;
print G qq~set xlabel "sqrtS"; set ylabel "counts"\n~;
$q__ = '';
for $i (0..8) {
  $x = 10**$i;
  $q__.= qq[\"10\^\{$i\}\" $x,];
  for $c (2..9) {
    $x = 10**$i*$c;
    $q__.= qq["" $x,];
  }
}
chop($q__);
print G qq~set ytics ($q__); set logscale y\n~;
print G qq~set boxwidth 0.9 relative\n~;
print G qq~set label "$pc\% at >2" at 1.4,.2*$ymax\n~;
print G qq~plot "results/$tag-$band\__sqrtS_hist.dat" with boxes fs solid 0.75\n~;
#
print G qq~unset label\n~;
print G qq~set origin 1,1.4\n~;
print G qq~set title "coverage histogram [case: ~.$tt.qq~ band: $band]"\n~;
print G qq~set xrange [0:10]; set xtics 1\n~;
($cs, $ymax, $pc) = split ' ', `head -n 1 results/$tag-$band\__neff_hist.dat`;
print G qq~set yrange [0.9:1.1*$ymax]; set logscale y\n~;
print G qq~set xlabel "effective coverage"; set ylabel "counts"\n~;
$q__ = '';
for $i (0..8) {
  $x = 10**$i;
  $q__.= qq[\"10\^\{$i\}\" $x,];
  for $c (2..9) {
    $x = 10**$i*$c;
    $q__.= qq["" $x,];
  }
}
chop($q__);
print G qq~set ytics ($q__); set logscale y\n~;
print G qq~set boxwidth 0.9 relative\n~;
print G qq~set label "$pc\% at >10" at 8,.2*$ymax\n~;
print G qq~plot "results/$tag-$band\__neff_hist.dat" with boxes fs solid 0.75\n~;
print G qq~unset multiplot\n~;
close G;
