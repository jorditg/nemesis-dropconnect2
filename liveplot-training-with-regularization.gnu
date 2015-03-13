set title 'NN training error evolution' # plot title
set xlabel 'Epoch'  # x-axis label
set ylabel 'Error'  # y-axis label

set xrange [0:*]    # fixed lower limit, dynamic upperlimit
set yrange [0:*]    # fixed lower limit, dynamic upperlimit
plot "training.dat" using 1:2 with lines    # training data (x->col1, y->col2 )
plot "training.dat" using 1:3 with lines    # test data (x->col1, y->col3 )
pause 5     # update every 5 seconds
reread      # reread file and replot
