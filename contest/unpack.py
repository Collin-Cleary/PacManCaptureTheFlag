import os, cPickle, sys

if len(sys.argv) != 3: 
  print 'Usage: %s stats_file team_name' % sys.argv[0]
  print 'Unpacks the stats file of a server into a bunch of replay files.'
  if len(sys.argv) == 2: 
    d = cPickle.load(open(sys.argv[1]))
    print 'Team names:', d.keys()
  sys.exit(2)

d = cPickle.load(open(sys.argv[1]))
user = sys.argv[2]
k = 0
print 'Unpacking games for', user
for g, w in d[user]['gameHistory']:
    k += 1
    t = {'layout': g.state.data.layout, 'agents': g.agents, 'actions': g.moveHistory, 'length': g.length}
    fname = 'replay_' + user + '_' + str(k)
    print 'Game:', fname
    cPickle.dump(t,file(fname, 'w'))