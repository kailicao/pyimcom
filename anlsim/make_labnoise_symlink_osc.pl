# This is the script I am using on OSC to make the symbolic links for
# the lab noise on OSC. It could be copied/modified for other platforms.
#
# Right now this is a placeholder because I copied over one noise exposure
# that is getting used over and over again (Noise001). I will replace this
# before the production run.
#
# -C.H.

$indir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/';
$linkdir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise/';

@files = split ' ', `ls $indir`;

@count_sca = (0)x18;

for $infile (@files) {

  $infile =~ m/Roman_WAS_simple_model_([A-Z0-9]+)_(\d+)_(\d+)\.fits/;
  $band = $1; $obs = $2; $sca = $3;

  for $i (0..17) {print (sprintf "%3d", $count_sca[$i]);}
  print "\n        band $band; obs $obs; sca $sca";

  # make target
  $count_sca[$sca-1]++;
  $target = sprintf "/fs/scratch/PCON0003/cond0007/labnoise/Noise%03d_SCA%02d_slopes_refcor.fits", $count_sca[$i], $sca;
  $link = $linkdir.(sprintf "slope_%d_%d.fits", $obs, $sca);
  print "\n $link -> $target\n\n";
  system "ln -s $target $link";
}

