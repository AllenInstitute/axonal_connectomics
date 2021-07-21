arglist = getArgument();
args = split(arglist, '#');

print("Running analysis with arguments:");
for (i = 0; i < args.length; i++)
    print(args[i]);

num_prints = parseInt(args[1]);
for (i = 0; i < num_prints; i++)
    print(args[0])