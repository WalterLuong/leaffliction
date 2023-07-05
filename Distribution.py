# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Distribution.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: wluong <wluong@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/07/05 11:30:55 by wluong            #+#    #+#              #
#    Updated: 2023/07/05 14:03:47 by wluong           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from pathlib import Path
import sys
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Path folder is missing.")
        exit(1)
    pathname = Path("./data/images/", sys.argv[1])

    try:
        assert pathname.is_dir(), "Invalid path."
    except AssertionError as e:
        print(e)
        sys.exit(1)

    # Data and labels
    colors = ['cornflowerblue', 'crimson',
              'green', 'pink', 'cyan', 'pink', 'yellow']
    classes = [str(Path(pathname, x).stem)
               for x in pathname.iterdir() if x.is_dir()]
    distribution = [(len(list(Path(pathname, classe).iterdir())))
                    for classe in classes]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
    fig1.tight_layout(pad=15.0)
    fig1.suptitle(Path(sys.argv[1]).stem + " class distribution")

    # Pie distribution
    ax1.pie(distribution, labels=classes,
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')

    # Histogram distribution
    plt.bar(classes, height=distribution, color=colors, edgecolor='black')
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='gray', linestyle='dashed')
    plt.show()
