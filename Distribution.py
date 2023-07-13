# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Distribution.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: wluong <wluong@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/07/05 11:30:55 by wluong            #+#    #+#              #
#    Updated: 2023/07/12 15:42:46 by wluong           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import warnings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] \
        in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("Path folder is missing.")
        exit(1)
    pathname = Path(sys.argv[1])

    try:
        if not pathname.is_dir():
            raise Exception("Path is not a directory.")
    except Exception as e:
        logger.error(e)
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
