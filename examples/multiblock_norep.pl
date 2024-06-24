# Script to run several configuration files in parallel.
# Useful for running on a multi-core machine.
#
# It has to be called *from one directory up* (so from pyimcom/ need to do a cd .. before running)
# This version: Developed by Chris Hirata; customized by Kaili Cao for Paper III production runs.

use POSIX;

($band, $kernel, $NB, $start, $Nr, $Nc) = @ARGV;
$config = "paper3_configs/${band}_${kernel}_config.json";
$outstem = "/fs/scratch/PAS2733/kailicao/paper3_production/${band}_${kernel}/${band}_${kernel}";

$N = $Nc*$Nr;
$p = 739;
$cnt = 0;
print " \$k  \$x \$y \$block\n";
for $k (0..$N-1) {
  $xy = (($start+$k)*$p)%($NB**2);
  $y = $xy%$NB;
  $x = int(($xy-$y)/$NB);

  $block_ = $NB*$x + $y;
  $suffix_ = sprintf "_%02d_%02d", $x, $y;
  $outfile = "$outstem$suffix_.out";
  if (-e $outfile) {
      print sprintf "%4d %02d %02d %4d -> exists\n", $k, $x, $y, $block_;
  } else {
      print sprintf "%4d %02d %02d %4d\n", $k, $x, $y, $block_;
      $block[$cnt] = $block_;
      $suffix[$cnt] = $suffix_;
      $cnt++;
  }
}
print "$cnt run(s)\n";
for $k ($cnt..$N-1) {
  $block[$k] = -1;
  $suffix[$k] = "";
}
print "\n";

my $i=0;
for $i (0..$Nr-1) {
  my $pid=fork;
  if (not defined $pid) {
    print STDERR "Error: fork failed\n";
    exit;
  }
  if (not $pid) {
    for $j (0..$Nc-1) {
      $ii = $i+$j*$Nr;
      if ($block[$ii] >= 0) {
        $outfile = "$outstem$suffix[$ii].out";
        $command1 = "date > $outfile; echo \"\" >> $outfile";
        $command2 = "lscpu >> $outfile; echo \"\" >> $outfile";
        $command3 = "python3 run_pyimcom_oscrun1.py $config $block[$ii] >> $outfile";
        # print "$command1\n";
        # print "$command2\n";
        print strftime "%Y-%m-%d %H:%M:%S\n", localtime time;
        print "$command3\n\n";
        system $command1;
        system $command2;
        system $command3;
      }
    }
    exit;
  }
}

# Wait for children
my $k;
for $k (1..$N) {wait();}
print strftime "%Y-%m-%d %H:%M:%S\n", localtime time;
