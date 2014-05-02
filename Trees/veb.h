#pragma once
#include "ct.h"
#include "ct_error.h"
#include "ct_geometry.h"
#include "ct_tree.h"
#include "ct_memory.h"

static int fls(int f)
{
    int order;
    for (order = 0; f != 0;f >>= 1, order++);
    return order;
}

static int ilog2(int f)
{
    return fls(f) - 1;
}

static boolean is_power_of_two(int f)
{
    return (f & (f-1)) == 0;
}

static int hyperfloor(int f)
{
    return 1 << (fls(f) - 1);
}

static int hyperceil(int f)
{
    return 1 << fls(f-1);
}

/*
* Given the BFS numbering of a node, compute its vEB position.
*
* BFS number is in the range of 1..#nodes.
*/
static int bfs_to_veb(int bfs_number, int height)
{
    int split;
    int top_height, bottom_height;
    int depth;
    int subtree_depth, subtree_root, num_subtrees;
    int toptree_size, subtree_size;
    int mask;
    int prior_length;

    /* if this is a size-3 tree, bfs number is sufficient */
    if (height <= 2)
        return bfs_number;

    /* depth is level of the specific node */
    depth = ilog2(bfs_number);

    /* the vEB layout recursively splits the tree in half */
    split = hyperceil((height + 1) / 2);
    bottom_height = split;
    top_height = height - bottom_height;

    /* node is located in top half - recurse */
    if (depth < top_height)
        return bfs_to_veb(bfs_number, top_height);

    /*
    * Each level adds another bit to the BFS number in the least
    * position. Thus we can find the subtree root by shifting off
    * depth - top_height rightmost bits.
    */
    subtree_depth = depth - top_height;
    subtree_root = bfs_number >> subtree_depth;

    /*
    * Similarly, the new bfs_number relative to subtree root has
    * the bit pattern representing the subtree root replaced with
    * 1 since it is the new root. This is equivalent to
    * bfs' = bfs / sr + bfs % sr.
    */

    /* mask off common bits */
    num_subtrees = 1 << top_height;
    bfs_number &= (1 << subtree_depth) - 1;

    /* replace it with one */
    bfs_number |= 1 << subtree_depth;

    /*
    * Now we need to count all the nodes before this one, then the
    * position within this subtree. The number of siblings before
    * this subtree root in the layout is the bottom k-1 bits of the
    * subtree root.
    */
    subtree_size = (1 << bottom_height) - 1;
    toptree_size = (1 << top_height) - 1;

    prior_length = toptree_size +
        (subtree_root & (num_subtrees - 1)) * subtree_size;

    return prior_length + bfs_to_veb(bfs_number, bottom_height);
}


// int main(void)
// {
//     const int h = 4;
//     const int size = (1 << h) - 1;
//     for(int i = 0; i < size; ++i)
//     {
//         ct_printf("%d ", bfs_to_veb(1+i, h));
//     }
//     ct_printf("\n");
//     return 0;
// }