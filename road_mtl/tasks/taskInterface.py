
# import t


class TaskInterface:

    def encode(self, mat):
        """
        we need resnet.
        create
        """

        resnet = bb.get_backbone(arch="resnet18", n_frames=args.SEQ_LEN)
        clip.unsqueeze_(0)

        print(resnet(clip))

        pass