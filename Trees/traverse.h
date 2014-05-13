#pragma once
#include "ct.h"

void TraverseDepthFirst
    (
    ICTTreeNode* root, 
    OnNodeTraverse callBack
    );

void TraverseBreathFirst
    (
    ICTTreeNode* root, 
    OnNodeTraverse callBack
    );

void TraverseVEB
    (
    ICTTreeNode* root, 
    OnNodeTraverse callBack
    );