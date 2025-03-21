####################################################
Utilities used in the OpenUniverse 2024 PyIMCOM Runs
####################################################

The scripts here are not part of the "standard" PyIMCOM; rather they are auxiliary scripts to convert OpenUniverse 2024 products to the ``anlsim`` format.

**************
PSF conversion
**************

The psf conversion script is ``genpsf.py``. The input format that PyIMCOM expects is a Legendre polynomial cube, which is a 3D FITS image HDU. In brief, the Legendre polynomials are defined in terms of the re-scaled SCA coordinates ``u=(x-2044.5)/2044, v=(y-2044.5)/2044``. There are ``(PORDER+1)**2`` coefficients (here ``PORDER=1``), in the order ``P_0(u)P_0(v), P_1(u)P_0(v), P_0(u)P_1(v), P_1(u)P_1(v)``. The ``-2`` and ``-1`` axes are then the y and x axes of that component of the PSF.

The ``genpsf.py`` script calls ``roman_imsim.utis.roman_utils`` to generate the PSF (which itself is linearly interpolated from the 4 corners) and convert this into the Legendre polynomial basis. It takes the observation ID number as a single command-line input.

The output PSF is directed to a file in the ``psf/`` subdirectory (which you need to make beforehand), in the format: ``'psf/psf_polyfit_{:d}.fits'.format(id)``.

For the Ohio Supercomputer Center (OSC) OpenUniverse2024 PyIMCOM runs, we used the following perl script (``run.pl``) to generate a bunch of PSFs from observation IDs ``$istart`` to ``$iend``, inclusive::

  ($istart, $iend) = @ARGV;

  print "== RUNNING $istart .. $iend ==\n";

  for $i ($istart..$iend) {
    print "$i --> ";
    system('date');
    @data = split ' ', `python chooseruns.py $i`;
    print "$data[0] @ $data[1] $data[2]\n";
    if ($data[0]<1.5) {
      system "python genpsf.py $i";
      system "mv psf/psf_polyfit_$i.fits /fs/scratch/PCON0003/cond0007/anl-run-in-prod/psf/psf_polyfit_$i.fits";
    } else {
      print "    skip\n";
    }
    print "finish --> ";
    system('date');
  }

(the destination of the ``mv`` is something you will want to change based on your platform). We ran all of the jobs by using the following perl script that submits jobs::

  $nrun = 16956;

  $ct = 0;
  $istart = 0;
  $ngrp = 300;
  while ($istart<$nrun) {

    $iend = $istart+$ngrp-1;
    if ($iend>=$nrun) {$iend = $nrun-1;}

    open(OUT, (sprintf ">PSF-%03d.job", $ct));
    print OUT (sprintf "#SBATCH --job-name=PSF-%03d\n", $ct);
    print OUT "#SBATCH --account=PCON0003\n";
    print OUT "#SBATCH --time=120:00:00\n";
    print OUT "#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2\n";
    print OUT "cd \$SLURM_SUBMIT_DIR\n";
    print OUT "perl run.pl $istart $iend\n";
    close OUT;

    $command = sprintf ("sbatch PSF-%03d.job", $ct);
    print("$command\n");
    system $command;

    $istart += $ngrp;
    $ct++;
  }

(Again, the slurm directives will be different on your platform, and you may choose different sizes.)
